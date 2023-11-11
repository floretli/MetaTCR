from metatcr.encoders.build_graph import update_graph_config, build_corr_graph, load_pkfile
import configargparse
import os

parser = configargparse.ArgumentParser()

parser.add_argument('--clst_num', type=int, default=96, help='Number of clusters for building global graph')
parser.add_argument('--file_list', type=str, default='./data/example_MDA-N.list', help='File or file list for processing')
parser.add_argument('--out_dir', type=str, default='./results06/co_occurr_graph', help='Output directory for processed data')
parser.add_argument('--db_cut_size', type=int, default=5000, help='How many sequence in each data file to process pre-clustering or frequency count')
parser.add_argument('--file_cut_size', type=int, default=500, help='How many files should be selected from the filelist')
parser.add_argument('--centroids_dir', type=str, default='./results06/data_analysis', help='Output directory for processed data')
parser.add_argument('--keep_edge_rate', type=float, default=0.5, help='Generated edge rate for building global graph')
parser.add_argument('--tcr_type', type=str, default='full_seq', choices=['full_length', 'cdr3'],
                    help='TCR type: full_length or cdr3')
parser.add_argument('--freq_col', type=str, default='frequencyCount (%)', help='Column name for frequency count')
parser.add_argument('--tcr_col', type=str, default='full_seq',
                    help='Column name for full length TCR sequence or amino acid')

args = parser.parse_args()
update_graph_config(args)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

centroids = load_pkfile(os.path.join(args.centroids_dir, str(args.clst_num) + '_best_centers.pk'))
# ## build global graph and get cluster centroid
# t0 = time.time()
# build_co_occurr_graph( args.file_list, centroids, args.out_dir , args.keep_edge_rate)
# t1= time.time()
# print('Time for building global graph: ', t1-t0)


freq_mtx = load_pkfile(os.path.join("./results06/co_occurr_graph", "freq_mtx_" + str(args.clst_num) + ".pk"))

build_corr_graph( args.file_list, centroids, "./results06/corr_graph02" , 0.2, freq_mtx)