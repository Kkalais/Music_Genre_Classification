import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns 

def create_numerics(data):
    #Get nominal columns
    nominal_cols = data.select_dtypes(include='object').columns.tolist()

    genres_with_ids = pd.DataFrame(data.genre_top)

	#Turn nominal to numeric 
    for nom in nominal_cols:
        enc = LabelEncoder()
        enc.fit(data[nom])
        data[nom]=enc.transform(data[nom])
        
    genres_with_ids["ids"] = pd.Series(data.genre_top)

    return (genres_with_ids.drop_duplicates(),data)

def prepare_data(plot_flag):
    
    data_without_labels = pd.read_csv("features.csv",low_memory=False)
    selected_features=[0,1,2,3,4,5,6,7,8,21,22,23,29,30,31,32,33,34,35,36,37,38,39,40]
    
    data_without_labels = data_without_labels.iloc[:,selected_features]
    data_without_labels['labels'] = pd.Series()
  
    tracks_with_labels = pd.DataFrame(pd.read_csv("tracks.csv",low_memory=False),columns=['track_id','genre_top'])
    tracks_with_labels = tracks_with_labels.dropna()
    genres_with_ids,tracks_with_labels = create_numerics(tracks_with_labels)
    
    track_id_A = np.array(data_without_labels['track_id'])
    track_id_B = np.array(tracks_with_labels['track_id'])
    
    for i in track_id_A:
        if i in track_id_B:
            indexA = int(np.where(track_id_A==i)[0])
            indexB = int(np.where(track_id_B==i)[0])
            data_without_labels.at[indexA,'labels'] = tracks_with_labels.iloc[indexB][1]

    #delete row instances with nan value at labels
    data_without_labels = data_without_labels.dropna().drop(["track_id"],axis=1)
    
    # Create correlation matrix
    corr_matrix = data_without_labels.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
    
    # Drop features 
    data_without_labels = data_without_labels.drop(data_without_labels[to_drop], axis=1)
    
    data_without_labels = shuffle(data_without_labels,random_state=42)
    
    y = pd.DataFrame(data_without_labels["labels"]).values
    X = data_without_labels.drop(["labels"],axis=1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    if plot_flag == 1:
        #plot the input data
        plot_data(data_without_labels,genres_with_ids)
    
        #plot the correlated features
        sns.heatmap(
             corr_matrix, 
             vmin=-1, vmax=1, center=0,
             cmap=sns.diverging_palette(20, 220, n=200),
             square=True
        )
        plt.title("Features correlation")
        plt.show()
    
    return(X_train,X_test,y_train,y_test,genres_with_ids)

def plot_data(data,genres_with_ids):
    
    genres_with_duplicates = list(data["labels"])
    genres = list(genres_with_ids["ids"])
    genres_labels = list(genres_with_ids["genre_top"])
    songs_per_genre = [genres_with_duplicates.count(genre) for genre in genres]
    y_pos = np.arange(len(genres_labels))
        
    plt.barh(y_pos,songs_per_genre, align='center', alpha=0.5)
    plt.yticks(y_pos,genres_labels)
    plt.xlabel('Usage')
    plt.title('Number of Songs per Genre')
    plt.xlim(0,1200)
    plt.show()