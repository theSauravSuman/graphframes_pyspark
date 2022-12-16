from GraphFrame import *
import os

class EigenvectorCentrality(GraphFrame):

    def __init__(self, edges: DataFrame, type: str = "undirected", weighted: bool = False, defaultWeight: int = 1) -> None:
        super().__init__(edges, type, weighted, defaultWeight)

    def run(self, precision = 4, checkpointInterval = 2) -> DataFrame:  # currently works with undirected only
        shouldCheckpoint = checkpointInterval > 0
        if shouldCheckpoint:
            checkpointDir = self.sparkSession.sparkContext.getCheckpointDir()
            if not checkpointDir:
                raise Exception("Please set checkpoint directory using sc.setCheckpointDir().")
            else:
                checkpointDir = os.path.join(checkpointDir, "checkpoint.dat")
        else:
            print(f"Checkpointing is disabled because checkpointInterval={checkpointInterval}. It may lead to system crash!")
        eigenvectorScores = self.nodes.withColumn("eigenvalue", py_fn.lit(1.0))
        lastEigenvalue = 1
        iter = 0
        converged = False
        while not converged:
            sourceInterScore = self.graph.join(eigenvectorScores, self.graph.target == eigenvectorScores.node, "left")
            sourceInterScore = sourceInterScore.groupBy(py_fn.col("source").alias("node")).agg(py_fn.sum("eigenvalue").alias("eigenvalue"))
            targetInterScore = self.graph.join(eigenvectorScores, self.graph.source == eigenvectorScores.node, "left")
            targetInterScore = targetInterScore.groupBy(py_fn.col("target").alias("node")).agg(py_fn.sum("eigenvalue").alias("eigenvalue"))
            eigenvectorScores = sourceInterScore.union(targetInterScore).groupBy("node").agg(py_fn.sum("eigenvalue").alias("eigenvalue"))
            eigenvectorScores = eigenvectorScores.crossJoin(eigenvectorScores.groupBy().agg(py_fn.max("eigenvalue").alias("max_eigenvalue")))
            eigenvectorScores = eigenvectorScores.select("node", "max_eigenvalue", (py_fn.col("eigenvalue")/py_fn.col("max_eigenvalue")).alias("eigenvalue"))
            eigenvectorScores.persist(self.storageLevel)
            maxEigenvalue = float(eigenvectorScores.limit(1).select("max_eigenvalue").collect()[0].__getitem__("max_eigenvalue"))
            if round(lastEigenvalue, precision) == round(maxEigenvalue, precision):
                converged = True
                eigenvectorScores = eigenvectorScores.drop("max_eigenvalue")
                print(f"Eigenvector Centrality converged in {iter} iterations")
            else:
                iter += 1
                lastEigenvalue = maxEigenvalue
                if shouldCheckpoint and iter % checkpointInterval == 0:
                    eigenvectorScores.write.mode("overwrite").parquet(checkpointDir)
                    eigenvectorScores.unpersist()
                    eigenvectorScores = self.sparkSession.read.parquet(checkpointDir)
        return eigenvectorScores
