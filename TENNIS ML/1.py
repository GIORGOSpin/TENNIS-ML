import pandas as pd

df = pd.read_csv("atp_matches_2024.csv")

train = df[df["tourney_date"] < 20240925].copy()
test = df[df["tourney_date"] >= 20240925].copy()

print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
predictors = ["tourney_id","tourney_name","surface","draw_size","tourney_level","tourney_date","match_num","winner_id","winner_seed","winner_entry","winner_hand","winner_ht","winner_ioc","winner_age","loser_id","loser_seed","loser_entry","loser_name","loser_hand","loser_ht","loser_ioc","loser_age","score","best_of","round","minutes","w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced","l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced","winner_rank","winner_rank_points","loser_rank","loser_rank_points"]
target = "winner_name"

reg.fit(train[predictors],train["winner_name"])

predictions = reg.predict(test[predictions])



'''
na kanw clean ta data pou den exoun 
print("Rows with missing data before cleaning:", df.isnull().any(axis=1).sum())

df_clean = df.dropna()
print("Rows with missing data after cleaning:", df_clean.isnull().any(axis=1).sum())
print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)
'''



'''
kwdikas apo google_block

y = df['winner_name']
x = df.drop('winner_name', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
        random_state=100)



from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train , y_train) '''