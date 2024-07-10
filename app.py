from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv('games-features.csv')
cols_to_keep = ['ResponseID', 'ResponseName', 'GenreIsIndie', 'GenreIsAction', 'GenreIsAdventure', 'GenreIsCasual', 'GenreIsStrategy', 'GenreIsRPG', 'GenreIsSimulation', 'SteamSpyPlayersEstimate', 'AboutText']
df = df[cols_to_keep]

# Fill missing values
median_players = df['SteamSpyPlayersEstimate'].median()
df['SteamSpyPlayersEstimate'].fillna(median_players, inplace=True)
df['AboutText'].fillna('', inplace=True)

# Convert boolean columns to integers
bool_columns = ['GenreIsIndie', 'GenreIsAction', 'GenreIsAdventure', 'GenreIsCasual', 'GenreIsStrategy', 'GenreIsRPG', 'GenreIsSimulation']
df[bool_columns] = df[bool_columns].astype(int)

# Standardize 'SteamSpyPlayersEstimate'
scaler = StandardScaler()
df['SteamSpyPlayersEstimateScaled'] = scaler.fit_transform(df[['SteamSpyPlayersEstimate']])

# Create user-item matrix for collaborative filtering
user_item_matrix = df.pivot_table(index='ResponseID', columns='ResponseName', values='SteamSpyPlayersEstimateScaled', fill_value=0)

# Compute similarity matrices
item_similarity = cosine_similarity(user_item_matrix.T)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['AboutText'])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_by_game', methods=['POST'])
def recommend_by_game():
    data = request.json
    game_name = data.get('game_name').strip().lower()
    n = data.get('n', 10)
    
    available_games = [col.lower() for col in user_item_matrix.columns]
    
    if game_name not in available_games:
        print(f"Game '{game_name}' not found in user_item_matrix columns.")
        print(f"Available games: {available_games[:10]}")  # Print first 10 available games for debugging
        return jsonify([])
    
    game_idx = available_games.index(game_name)
    similar_indices = item_similarity[game_idx].argsort()[::-1][:n+1]  # Include the game itself
    similar_games = [(user_item_matrix.columns[i], item_similarity[game_idx, i]) for i in similar_indices if user_item_matrix.columns[i].lower() != game_name]
    
    print("Recommendations by game:", similar_games)
    return jsonify(similar_games)

@app.route('/recommend_by_description', methods=['POST'])
def recommend_by_description():
    data = request.json
    game_name = data.get('game_name').strip().lower()
    n = data.get('n', 10)
    
    available_games = df['ResponseName'].str.lower().values
    
    if game_name not in available_games:
        print(f"Game '{game_name}' not found in DataFrame.")
        print(f"Available games: {available_games[:10]}")  # Print first 10 available games for debugging
        return jsonify([])
    
    game_idx = df[df['ResponseName'].str.lower() == game_name].index[0]
    similar_indices = content_similarity[game_idx].argsort()[::-1][1:n+1]  # Exclude itself
    similar_games = df.iloc[similar_indices]['ResponseName'].values.tolist()
    
    print("Recommendations by description:", similar_games)
    return jsonify(similar_games)

@app.route('/show_graphs')
def show_graphs():
    # Create and save visualizations
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.countplot(y="ResponseName", data=df[df['SteamSpyPlayersEstimate'] > 10000].sort_values(by='SteamSpyPlayersEstimate', ascending=False).head(10))
    plt.title('Top 10 Popular Games by Player Estimates')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('graphs.html', plot_url=plot_url)


if __name__ == '__main__':
    print(df.head())  # Print sample data for debugging
    print(df.columns)  # Print column names for debugging
    print(user_item_matrix.head())  # Print user-item matrix for debugging
    print(item_similarity[:5, :5])  # Print small portion of item similarity matrix
    print(content_similarity[:5, :5])  # Print small portion of content similarity matrix
    
    app.run(debug=True)
