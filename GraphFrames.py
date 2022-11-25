# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. Getting Started

# COMMAND ----------

from graphframes import *

# COMMAND ----------

from pyspark import *
from pyspark.sql import *

# COMMAND ----------

spark = SparkSession.builder.appName('fun').getOrCreate()
spark

# COMMAND ----------

vertices = spark.createDataFrame([('1', 'Carter', 'Derrick', 50), 
                                  ('2', 'May', 'Derrick', 26),
                                 ('3', 'Mills', 'Jeff', 80),
                                  ('4', 'Hood', 'Robert', 65),
                                  ('5', 'Banks', 'Mike', 93),
                                 ('98', 'Berg', 'Tim', 28),
                                 ('99', 'Page', 'Allan', 16)],
                                 ['id', 'name', 'firstname', 'age'])

# COMMAND ----------

vertices

# COMMAND ----------

edges = spark.createDataFrame([('1', '2', 'friend'), 
                               ('2', '1', 'friend'),
                              ('3', '1', 'friend'),
                              ('1', '3', 'friend'),
                               ('2', '3', 'follows'),
                               ('3', '4', 'friend'),
                               ('4', '3', 'friend'),
                               ('5', '3', 'friend'),
                               ('3', '5', 'friend'),
                               ('4', '5', 'follows'),
                              ('98', '99', 'friend'),
                              ('99', '98', 'friend')],
                              ['src', 'dst', 'type'])

# COMMAND ----------

edges

# COMMAND ----------

g = GraphFrame(vertices, edges)

# COMMAND ----------

# Take a look at the DataFrames
g.vertices.show()
g.edges.show()

# COMMAND ----------

# Check the number of edges of each vertex
g.degrees.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Directed vs undirected edges

# COMMAND ----------

copy = edges
from pyspark.sql.functions import udf
@udf("string")
def to_undir(src, dst):
    if src >= dst:
        return 'Delete'
    else : 
        return 'Keep'
copy.withColumn('undir', to_undir(copy.src, copy.dst))\
.filter('undir == "Keep"').drop('undir').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Filtering and connected components

# COMMAND ----------

g.vertices.filter("age > 30").show()
g.inDegrees.filter("inDegree >= 2").sort("inDegree", ascending=False).show()
g.edges.filter('type == "friend"')

# COMMAND ----------

sc.setCheckpointDir('https://community.cloud.databricks.com/?o=291647158364918#folder/86354967488104')

# COMMAND ----------

g.connectedComponents().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Motif finding

# COMMAND ----------

g.find("(a)-[e]->(b); (b)-[e2]->(a)").show()

# COMMAND ----------

mutualFriends = g.find("(a)-[]->(b); (b)-[]->(c); (c)-[]->(b); (b)-[]->(a)")\
.dropDuplicates()

# COMMAND ----------

mutualFriends.filter('a.id == 2 and c.id == 3').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. TriangleCount and PageRank

# COMMAND ----------

g.triangleCount().show()

# COMMAND ----------

pr = g.pageRank(resetProbability=0.15, tol=0.01)
## look at the pagerank score for every vertex
pr.vertices.show()
## look at the weight of every edge
pr.edges.show()
