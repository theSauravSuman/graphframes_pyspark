import pyspark
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as py_fn
from pyspark.storagelevel import StorageLevel


class GraphFrame(object):

    def __init__(self, edges: DataFrame, clean_edges: bool = True, type: str = "undirected", weighted: bool = False, defaultWeight: int = 1, storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> None:
        self.sparkSession = edges.sparkSession
        self.sc = self.sparkSession.sparkContext
        self.type = type
        self.weighted = weighted
        self.defaultWeight = defaultWeight
        self.graph = edges
        self.storageLevel = storageLevel
        if self.weighted and "weight" not in self.graph.columns:
            self.graph = self.graph.withColumn("weight", 1)
        if clean_edges:
            self.clean_edges()
        self.nodes = self.graph.select(py_fn.col("source").alias("node")).union(self.graph.select("target")).distinct()
        self.graph.persist(self.storageLevel)
        self.nodes.persist(self.storageLevel)

    def clean_edges(self):
        """Utility Function to clean the graph edges with nulls and remove repeated edges. To be used only with Undirected type Graph"""
        if self.type == "undirected":
            split_pattern = "##"
            self.graph = self.graph.where(self.graph.source.isNotNull()).where(self.graph.target.isNotNull())
            self.graph = self.graph.na.fill(value = self.defaultWeight, subset=["weight"])
            self.graph = self.graph.withColumn("cat_nodes",
                py_fn.when(self.graph.source < self.graph.target,
                    py_fn.concat_ws(split_pattern, self.graph.source, self.graph.target))
                .when(self.graph.source > self.graph.target,
                    py_fn.concat_ws(split_pattern, self.graph.target, self.graph.source)))
            if self.weighted:
                self.graph = self.graph.groupBy("cat_nodes").agg(py_fn.sum("weight").alias("weight"))
            else:
                self.graph = self.graph.select("cat_nodes").distinct()
            split_nodes = py_fn.split(self.graph["cat_nodes"], split_pattern, 2)
            self.graph = self.graph.withColumn("source", split_nodes.getItem(0))\
                                .withColumn("target", split_nodes.getItem(1))\
                                .drop("cat_nodes")
        else:
            return "Cannot run helper function on Directed Graph as it might cause information loss"
