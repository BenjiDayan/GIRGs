import sys
sys.path.append('../nemo-eva/src/')
import graph_crawler
import os
os.environ["DATA_PATH"] = "/cluster/scratch/bdayan/girg_data/"

gc = graph_crawler.GraphCrawler()

gc._execute()

