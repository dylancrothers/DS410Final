#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
import torch.nn as nn
import torch
import wget


# In[2]:


ss=SparkSession.builder.appName("Final").getOrCreate()


# In[3]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[4]:


url = "https://raw.githubusercontent.com/dmmiller612/sparktorch/master/examples/mnist_train.csv"
destination_path = "mnist_train.csv"
wget.download(url, destination_path)


# In[5]:


df = ss.read.csv("mnist_train.csv", header=False, inferSchema=True)


# In[6]:


url = "https://raw.githubusercontent.com/dmmiller612/sparktorch/master/examples/cnn_network.py"
destination_path = "cnn_network.py"
wget.download(url, destination_path)


# In[7]:


from cnn_network import Net


# In[8]:


network = Net()


# In[9]:


torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.001
    )


# In[10]:


vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')


# In[11]:


spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=50,
        verbose=1,
        validationPct=0.2,
        miniBatch=128
    )


# In[12]:


p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
p.write().overwrite().save('simple_cnn')


# In[13]:


loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_cnn'))


# In[14]:


predictions = loaded_pipeline.transform(df).persist()


# In[15]:


evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")


# In[17]:


accuracy = evaluator.evaluate(predictions)
#print("Train accuracy = %g" % accuracy)


# In[ ]:




