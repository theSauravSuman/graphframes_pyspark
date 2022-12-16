import pyspark
from pyspark.sql import SparkSession
from EigenvectorCentrality import EigenvectorCentrality

if __name__ == "__main__":
    ss = SparkSession.builder.appName("test").getOrCreate()
    edgeData = ss.read.csv("graph_data.csv", header=True)
    eigen_centrality = EigenvectorCentrality(edges=edgeData)
    eigen_centrality.sc.setCheckpointDir("checkpt")
    eigen_scores = eigen_centrality.run(precision=2)
    eigen_scores.show()
