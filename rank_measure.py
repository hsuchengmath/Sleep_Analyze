import numpy as np

def Rank(day,test_No,user_Id,dict_user_X_predict_Y):
    rank = []
    for i in range(test_No):
        rank.append([])
        for j in range(len(user_Id)):
            rank[i].append(dict_user_X_predict_Y[user_Id[j]][i].shape[0]-np.sum(dict_user_X_predict_Y[user_Id[j]][i])+1)
    return rank[day]

def Rank_sort(True_data,Predicted_data):

    from scipy.stats import rankdata

    b = sorted([(True_data[i],Predicted_data[i]) for i in range(len(True_data))])
    rank = [b[i][1] for i in range(len(b))]
    #rank = rankdata(b_1, method='dense')
    return rank

def Kendall_Rank_Coefficicent(Predicted_Rank):
    p_2 = []
    for i in range(len(Predicted_Rank)-1):
        sort_value = sorted(Predicted_Rank[i+1:])
        p_1 = 0
        for j in range(len(sort_value)):
            if Predicted_Rank[i]<sort_value[j]:
                p_1 = p_1+1
        p_2.append(p_1)

    p = sum(p_2)
    tou = ((4*p)/((len(Predicted_Rank))*(len(Predicted_Rank)-1)))-1
    return tou

def Spesrson_Rank_Coefficicent(Predicted_Rank):
    True_Rank = [i+1 for i in range(len(Predicted_Rank))]
    p = sum([(True_Rank[i]-Predicted_Rank[i])*(True_Rank[i]-Predicted_Rank[i]) for i in range(len(Predicted_Rank))])
    n = len(Predicted_Rank)
    tou = 1 - 6*p/(n*(n*n-1)) 
    return tou

def NDCG_Function(rank):
    rank_score = [max(rank)-rank[i]+1 for i in range(len(rank))]
    rank_score_sorted = sorted(rank_score,reverse=True)

    DCG = sum([((2**rank_score[i])-1)/(np.log2(i+2)) for i in range(len(rank_score))])
    IDCG  = sum([((2**rank_score_sorted[i])-1)/(np.log2(i+2)) for i in range(len(rank_score_sorted))])
    ncdg = DCG/IDCG
    return ncdg


def Precision_k(rank,k):
    p1 = [0]
    for i in range(k+1):
        if rank[i] in [j+1 for j in range(i+1)]:
            p1[i] = p1[i]+1
            if i != len(rank)-1:
                p1.append(p1[i])
        else:
            p1.append(p1[i])
    p1 = p1[:len(p1)-2]
    p2 = []
    for i in range(len(p1)):
        p2.append(p1[i]/(i+1))
    return p2





