#python3 convert_all.py citeseer      False |& tee ./logs/citeseer_convert.log  #1.3m
#python3 convert_all.py cora_ml      False |& tee ./logs/cora_ml_convert.log    #1.5m
#python3 convert_all.py pubmed       False |& tee ./logs/pubmed_convert.log     #8.5m
#python3 convert_all.py cora_full    False |& tee ./logs/cora_full_convert.log  #10.6m
#python3 convert_all.py reddit       False |& tee ./logs/reddit_convert.log      #1,38G

#python3 convert_all.py ogbn-products    True |& tee ./logs/ogbn-products_convert.log   #1,5 GB
python3 convert_all.py yelp             True |& tee ./logs/yelp_convert.log            #2,2 GB
#python3 convert_all.py amazon           True |& tee ./logs/amazon_convert.log          #3,7 GB

#python3 convert_all.py reddit           True |& tee ./logs/amazon_convert.log          #1,2 GB
