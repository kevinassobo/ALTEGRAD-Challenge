import os 
import sys



# Paths where we will save the datasets
# -------------------------------------
DATA_DIR = "../data"

# NOTE : uncomment if you are using google collab and change the data path 
# import sys
# if 'google.colab' in sys.modules:
#     DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/ALTeGraD/Projet/data"

#     from google.colab import drive
#     drive.mount('/content/drive')
  
INITIAL_DATA_DIR = os.path.join(DATA_DIR, 'initial_data')
AUTHORS_PROCESSED_DIR = os.path.join(DATA_DIR, 'authors_processed')
AUTHORS_CITATIONS_DIR = os.path.join(DATA_DIR, 'authors_citations')
GRAPHS_FEATS_DIR = os.path.join(DATA_DIR, 'graphs_features')
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

os.makedirs(INITIAL_DATA_DIR, exist_ok=True)
os.makedirs(AUTHORS_PROCESSED_DIR, exist_ok=True)
os.makedirs(AUTHORS_CITATIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_FEATS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Initial data directory
# ----------------------
ABSTRACTS_PATH = os.path.join(INITIAL_DATA_DIR, 'abstracts.txt')
AUTHORS_PATH = os.path.join(INITIAL_DATA_DIR, 'authors.txt')
FULL_GRAPH_EDGELIST_PATH = os.path.join(INITIAL_DATA_DIR, 'edgelist.txt')
CHALLENGE_PAIRS_PATH = os.path.join(INITIAL_DATA_DIR, 'test.txt')

# Authors processed directory
# ---------------------------
AUTHCOLL_EDGELIST_PATH = os.path.join(AUTHORS_PROCESSED_DIR, 'author_collab_edgelist.txt')
PAPER_2_AUTHORS_ID_PATH = os.path.join(AUTHORS_PROCESSED_DIR, 'paper_2_authors_id.txt')
AUTHOR_ID_2_PAPERS_PATH = os.path.join(AUTHORS_PROCESSED_DIR, 'author_id_2_papers.txt')
ID_2_AUTHOR_PATH = os.path.join(AUTHORS_PROCESSED_DIR, 'id2author.txt')

# Authors citations directory
# ---------------------------
FULL_AUTHCIT_EDGELIST_PATH = os.path.join(AUTHORS_CITATIONS_DIR, 'authcit_edgelist.txt')
TRAIN_AUTHCIT_EDGELIST_PATH = os.path.join(AUTHORS_CITATIONS_DIR, 'train_authcit_edgelist.txt')
TEST_AUTHCIT_EDGELIST_PATH = os.path.join(AUTHORS_CITATIONS_DIR, 'test_authcit_edgelist.txt')

# graphs features directory
# ---------------------------
FULL_GRAPH_BET_CENT_PATH = os.path.join(GRAPHS_FEATS_DIR, 'full_graph_bet_cen.pkl')
TRAIN_GRAPH_BET_CENT_PATH = os.path.join(GRAPHS_FEATS_DIR, 'train_graph_bet_cen.pkl')
TEST_GRAPH_BET_CENT_PATH = os.path.join(GRAPHS_FEATS_DIR, 'test_graph_bet_cen.pkl')

FULL_GRAPH_CLO_CENT_PATH = os.path.join(GRAPHS_FEATS_DIR, 'full_graph_clo_cen.pkl')
TRAIN_GRAPH_CLO_CENT_PATH = os.path.join(GRAPHS_FEATS_DIR, 'train_graph_clo_cen.pkl')
TEST_GRAPH_CLO_CENT_PATH = os.path.join(GRAPHS_FEATS_DIR, 'test_graph_clo_cen.pkl')

FULL_GRAPH_GAE_EMB_PATH = os.path.join(GRAPHS_FEATS_DIR, 'full_graph_gae_emb.pkl')
TRAIN_GRAPH_GAE_EMB_PATH = os.path.join(GRAPHS_FEATS_DIR, 'train_graph_gae_emb.pkl')
TEST_GRAPH_GAE_EMB_PATH = os.path.join(GRAPHS_FEATS_DIR, 'test_graph_gae_emb.pkl')

# Model directory
# ---------------
NODE2VEC_TRAIN_PATH  = os.path.join(MODELS_DIR, 'node2vec_train_graph.nodevectors')
NODE2VEC_TEST_PATH = os.path.join(MODELS_DIR, 'node2vec_test_graph.nodevectors')
NODE2VEC_FULL_GRAPH_PATH = os.path.join(MODELS_DIR, 'node2vec_full_graph.nodevectors')

DOC2VEC_PATH = os.path.join(MODELS_DIR, "doc2vec_dm_64.model")

# Dataset directory
# -----------------
CHALLENGE_FEATS_PATH = os.path.join(DATASETS_DIR, 'challenge_feats.csv')

# Intermediate storage (for non final datasets)
STORAGE_DIR = os.path.join(DATASETS_DIR, 'storage')
os.makedirs(STORAGE_DIR, exist_ok=True)

STORAGE_STAGE1_PATH = os.path.join(STORAGE_DIR, 'stage_1.h5')
STORAGE_STAGE2_PATH = os.path.join(STORAGE_DIR, 'stage_2.h5')

# Train data directory
TRAIN_DIR = os.path.join(DATASETS_DIR, 'train_data')
os.makedirs(TRAIN_DIR, exist_ok=True)

TRAIN_EDGELIST_PATH  = os.path.join(TRAIN_DIR, 'train_graph_edgelist.txt')
TRAIN_PAIRS_PATH  = os.path.join(TRAIN_DIR, 'train_pairs.csv')
TRAIN_TARGET_PATH  = os.path.join(TRAIN_DIR, 'train_target.csv')
TRAIN_FEATS_PATH = os.path.join(TRAIN_DIR, 'train_feats.csv')

# Test data directory
TEST_DIR = os.path.join(DATASETS_DIR, 'test_data')
os.makedirs(TEST_DIR, exist_ok=True)

TEST_EDGELIST_PATH  = os.path.join(TEST_DIR, 'test_graph_edgelist.txt')
TEST_PAIRS_PATH  = os.path.join(TEST_DIR, 'test_pairs.csv')
TEST_TARGET_PATH  = os.path.join(TEST_DIR, 'test_target.csv')
TEST_FEATS_PATH = os.path.join(TEST_DIR, 'test_feats.csv')


