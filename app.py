import streamlit as st
import pandas as pd
import pickle
import requests
import math
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load vectorizer, tfidf matrix, and data
@st.cache_data
def load_files():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('tfidf_matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)
    with open('output_with_lat_lon.pkl', 'rb') as file:
        data = pickle.load(file)
    return vectorizer, tfidf_matrix, data

vectorizer, tfidf_matrix, data = load_files()

# Geocoding function to get latitude and longitude
def get_lat_lon_from_address(address, access_token):
    try:
        url = f"https://us1.locationiq.com/v1/search.php?key={access_token}&q={address}&format=json"
        response = requests.get(url)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        return None, None
    return None, None

# Haversine formula to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Filter restaurants by location
def filter_by_location(user_lat, user_lon, restaurants, radius=7):
    return restaurants[
        restaurants.apply(lambda row: haversine_distance(user_lat, user_lon, float(row['latitude']), float(row['longitude'])) <= radius, axis=1)
    ]

def recommend_restaurants_by_address(user_lat, user_lon, cuisine=None, price_for_two=None, planning_for=None, liked_restaurant=None, rating=None, dish=None, radius=7, top_n=5):
    try:
        # Filter by location
        filtered_data = filter_by_location(user_lat, user_lon, data, radius)

        # Apply filters dynamically
        if rating is not None:
            filtered_data = filtered_data[filtered_data['rating'].between(rating, rating + 1.0)]
        
        if price_for_two is not None:
            filtered_data['price_for_two'] = pd.to_numeric(filtered_data['price_for_two'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['price_for_two'])
            filtered_data = filtered_data[filtered_data['price_for_two'] <= price_for_two]
        
        if cuisine:
            filtered_data = filtered_data[filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)]
        
        if planning_for:
            filtered_data = filtered_data[filtered_data['more_info'].str.contains(planning_for, case=False, na=False)]
        
        if dish:
            filtered_data = filtered_data[filtered_data['signature dish'].str.contains(dish, case=False, na=False)]

        if liked_restaurant:
            liked_index = data[data['name'].str.contains(liked_restaurant, case=False, na=False)].index
            
            if not liked_index.empty:
                similarity_scores = cosine_similarity(tfidf_matrix[liked_index[0]], tfidf_matrix[filtered_data.index]).flatten()
                filtered_data = filtered_data.copy()
                filtered_data.loc[:, 'weighted_score'] = similarity_scores * (filtered_data['rating'] / data['rating'].max())
                return filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)    
        
        return filtered_data.sort_values(by='rating', ascending=False).head(top_n)
    
    except Exception:
        return pd.DataFrame()


def recommend_restaurants_city(liked_restaurant=None, cuisine=None, budget=None, occasion=None, rating=None, dish=None, top_n=5):
    try:
        matches = data.copy()

        # Apply filters dynamically
        if cuisine:
            matches = matches[matches['cuisine'].str.contains(cuisine, case=False, na=False)]
        
        if budget is not None:
            matches['price_for_two'] = pd.to_numeric(matches['price_for_two'], errors='coerce')
            matches = matches.dropna(subset=['price_for_two'])
            matches = matches[matches['price_for_two'] <= budget]
        
        if occasion:
            matches = matches[matches['features'].str.contains(occasion, case=False, na=False)]
        
        if rating is not None:
            matches = matches[matches['rating'].between(rating, rating + 1.0)]
        
        if dish:
            matches = matches[matches['signature dish'].str.contains(dish, case=False, na=False)]
        
        if liked_restaurant:
            liked_index = data[data['name'].str.contains(liked_restaurant, case=False, na=False)].index
            
            if not liked_index.empty:
                similarity_scores = cosine_similarity(tfidf_matrix[liked_index[0]], tfidf_matrix[matches.index]).flatten()
                matches = matches.copy()
                matches.loc[:, 'weighted_score'] = similarity_scores * (matches['rating'] / data['rating'].max())
                return matches.sort_values(by='weighted_score', ascending=False).head(top_n)
        
        return matches.sort_values(by='rating', ascending=False).head(top_n)
    
    except Exception:
        return pd.DataFrame()


# Format the display of recommendations
# Format the display of recommendations
def display_recommendations(recommendations):
    if recommendations.empty:
        st.warning("No recommendations found based on your criteria.")
        return

    unique_recommendations = recommendations.drop_duplicates(subset=['name'])  # Ensure unique restaurants

    for _, row in unique_recommendations.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])  

            with col1:
                st.markdown(f"### 🍽️ {row['name']}")  
                st.markdown(f"**Cuisine:** {row['cuisine']}")
                st.markdown(f"**Price for Two:** ₹{row['price_for_two']}")

            with col2:
                rating_color = "🟢" if row['rating'] >= 4 else "🟡" if row['rating'] >= 3 else "🔴"
                st.markdown(f"**Rating:** {rating_color} {row['rating']}/5")
                st.markdown(f"**Features:** {row['features']}")
                st.markdown(f"**Adress:** {row['address']}")

            st.divider()


# Streamlit UI
data['features'] = data['more_info'].fillna('') + ',' + data['special features'].fillna('')


st.title("🍔 Restaurant Recommendation System")

# Sidebar input
st.sidebar.header("🔍 User Preferences")
address = st.sidebar.text_input("📍 Enter your location")
access_token = "pk.e28403f26ee75af55812430de27b810e"

cuisine = st.sidebar.text_input("🍽️ Preferred Cuisine (e.g., Indian, Chinese)")
price_for_two = st.sidebar.slider("💰 Budget for Two (₹)", 0, 3000)  
rating = st.sidebar.slider("⭐ Minimum Rating (0.0 to 5.0)", 0.0, 5.0, step=0.1)  
planning_for = st.sidebar.selectbox(  
    "🎉 Occasion / Planning for",  
    ['Live Music', 'Bar', 'Buffet', 'Outdoor Seating', 'Rooftop', 'Happy Hours',  
     'Karaoke', 'DJ', 'Romantic Dining', 'Sports Screening', 'Games', 'Pet Friendly',  
      'Fine Dining',  'Family Friendly', 'Theme Restaurant', 'Private Dining', 'Late Night Food'],  
    index=None  
)  


liked_restaurant = st.sidebar.text_input("👍 Liked Restaurant (optional)")
dish = st.sidebar.text_input("🍜 Enter a Dish Name (optional)")

if st.sidebar.button("🔎 Get Recommendations"):
    if address:
        user_lat, user_lon = get_lat_lon_from_address(address, access_token)
        if user_lat and user_lon:
            recommendations = recommend_restaurants_by_address(
                user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant, rating, dish
            )
            st.subheader("📍 Recommended Restaurants Nearby")
            display_recommendations(recommendations)
        else:
            st.error("⚠️ Invalid address. Please enter a valid location.")
    else:
        recommendations = recommend_restaurants_city(
            liked_restaurant, cuisine, price_for_two, planning_for, rating, dish
        )
        st.subheader("🌆 City-wide Recommendations")
        display_recommendations(recommendations)
