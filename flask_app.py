from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd # For any minor data manipulation if needed
import os
from dotenv import load_dotenv
import pymongo
from bson import ObjectId
from functools import wraps
# Import sentence-transformers for generating embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers package not installed. Vector search will not be available.")

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey" # For flash messages

# Add authentication configuration from .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash("Please log in to access this page", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Login/logout routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login authentication"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            next_page = request.args.get('next', url_for('index'))
            flash("Login successful!", "success")
            return redirect(next_page)
        else:
            flash("Invalid credentials. Please try again.", "danger")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.pop('logged_in', None)
    session.pop('username', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# --- Embedding Model ---
embedding_model = None

def load_embedding_model():
    """Load and return the sentence transformer embedding model"""
    global embedding_model
    
    if embedding_model is not None:
        return embedding_model
        
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
        
    try:
        # Use a small but effective model - all-MiniLM-L6-v2 is fast and produces 384-dim vectors
        model_name = 'all-MiniLM-L6-v2'
        print(f"Loading embedding model: {model_name}...")
        embedding_model = SentenceTransformer(model_name)
        print(f"Embedding model loaded successfully. Dimension: {embedding_model.get_sentence_embedding_dimension()}")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

# Try to load the model at startup
load_embedding_model()

# --- MongoDB Connection ---
def get_mongo_client():
    # Get MongoDB connection details from environment variables
    mongodb_uri_atlas = os.getenv("MONGODB_URI_ATLAS")
    ssl_cert_reqs = os.getenv("MONGODB_SSL_CERT_REQS", "CERT_REQUIRED")
    
    # Configure connection based on SSL settings
    if ssl_cert_reqs == "CERT_NONE":
        client = pymongo.MongoClient(
            mongodb_uri_atlas, 
            tlsInsecure=True
        )
    else:
        client = pymongo.MongoClient(mongodb_uri_atlas)
    return client

def get_db_collection(collection_name=None):
    """Get a MongoDB collection with optional different collection name"""
    try:
        client = get_mongo_client()
        db_name = os.getenv("DB_NAME", "tea_knowledge_graph")
        coll_name = collection_name or os.getenv("COLLECTION_NAME", "tea_businesses")
        return client[db_name][coll_name]
    except Exception as e:
        print(f"Error getting collection '{collection_name}': {e}")
        return None

# --- Helper Functions (MongoDB interactions) ---
def get_all_businesses_from_db(page=1, per_page=100):
    collection = get_db_collection()
    # Skip for pagination
    skip_count = (page - 1) * per_page
    
    # Get total count for pagination
    total_count = collection.count_documents({})
    
    # Get businesses with pagination
    businesses = list(collection.find({}).skip(skip_count).limit(per_page))
    
    # Convert ObjectId to string for JSON serialization
    for business in businesses:
        business['_id'] = str(business['_id'])
    
    return businesses, total_count

def get_business_by_id_from_db(business_id):
    collection = get_db_collection()
    try:
        # Convert string ID to ObjectId if it's in the right format
        if ObjectId.is_valid(business_id):
            business_id_obj = ObjectId(business_id)
            business = collection.find_one({"_id": business_id_obj})
        else:
            business = collection.find_one({"_id": business_id})
            
        if business:
            business['_id'] = str(business['_id'])
        return business
    except Exception as e:
        print(f"Error retrieving business: {e}")
        return None

def add_business_to_db(data):
    collection = get_db_collection()
    try:
        # Generate combined text for embedding (in a real app, we'd generate the embedding too)
        cols_for_embedding = ['Business Name', 'Type', 'Description', 'Location', 'Region']
        data['combined_text_for_embedding'] = ' '.join([str(data.get(col, '')) for col in cols_for_embedding])
        # In a real app, we'd also add the vector embedding here
        data['vector_embedding'] = [0.0] * int(os.getenv("EMBEDDING_DIM", 384))
        
        # Insert into MongoDB
        result = collection.insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error adding business: {e}")
        return None

def update_business_in_db(business_id, data):
    collection = get_db_collection()
    try:
        # Prepare ID for query
        if ObjectId.is_valid(business_id):
            business_id_obj = ObjectId(business_id)
        else:
            business_id_obj = business_id
            
        # Generate combined text for embedding
        cols_for_embedding = ['Business Name', 'Type', 'Description', 'Location', 'Region']
        data['combined_text_for_embedding'] = ' '.join([str(data.get(col, '')) for col in cols_for_embedding])
        # In a real app, we'd also update the vector embedding here
        
        # Update in MongoDB
        result = collection.update_one({"_id": business_id_obj}, {"$set": data})
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating business: {e}")
        return False

def delete_business_from_db(business_id):
    collection = get_db_collection()
    try:
        # Prepare ID for query
        if ObjectId.is_valid(business_id):
            business_id_obj = ObjectId(business_id)
        else:
            business_id_obj = business_id
            
        # Delete from MongoDB
        result = collection.delete_one({"_id": business_id_obj})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error deleting business: {e}")
        return False

def search_businesses_in_db(query, search_type="text"):
    collection = get_db_collection()
    vector_results = []
    try:
        if search_type == "vector" and os.getenv("EMBEDDING_DIM"):
            # Get the embedding model
            model = load_embedding_model()
            if not model:
                flash("Vector search requires the sentence-transformers package.", "warning")
                # Fall back to text search
                search_type = "text"
            else:
                # First, ensure vector index exists
                # Use the actual index name from MongoDB Atlas - use the name that already exists
                atlas_index_name = "vector_index"  # Use consistent name with what's in Atlas
                
                # Generate embedding for the search query
                try:
                    query_vector = model.encode(query).tolist()
                    embedding_dim = len(query_vector)
                    
                    print(f"Generated query embedding with dimension {embedding_dim} for '{query}'")
                    
                    # Check if vector dimensions match
                    if embedding_dim != int(os.getenv("EMBEDDING_DIM", 384)):
                        flash(f"Warning: Query embedding dimension ({embedding_dim}) doesn't match configured dimension ({os.getenv('EMBEDDING_DIM')}).", "warning")
                    
                    # Use MongoDB Atlas Vector Search with $vectorSearch
                    try:
                        print(f"Executing vector search with index '{atlas_index_name}' for query '{query}'")
                        
                        # Updated pipeline to use the actual index which includes additional fields
                        pipeline = [
                            {
                                "$vectorSearch": {
                                    "index": atlas_index_name,
                                    "path": "vector_embedding",
                                    "queryVector": query_vector,
                                    "numCandidates": 100,
                                    "limit": 100
                                }
                            },
                            {
                                "$project": {
                                    "_id": 1,
                                    "Business Name": 1,
                                    "Type": 1,
                                    "Description": 1,
                                    "Location": 1,
                                    "Website": 1,
                                    "Region": 1,
                                    "Contact Info": 1,
                                    "combined_text_for_embedding": 1,
                                    "score": {"$meta": "vectorSearchScore"}
                                }
                            }
                        ]
                        
                        vector_results = list(collection.aggregate(pipeline))
                        
                        # Convert ObjectId to string for each result
                        for result in vector_results:
                            result['_id'] = str(result['_id'])
                        
                        # If we have results, return them
                        if vector_results:
                            print(f"Vector search found {len(vector_results)} results for '{query}'")
                            return vector_results
                        else:
                            print(f"No vector search results for '{query}'")
                            flash("No results found with vector search. Trying text search instead.", "info")
                    except Exception as e:
                        print(f"Vector search error: {e}")
                        flash(f"Vector search failed: {str(e)}. Falling back to text search.", "warning")
                except Exception as e:
                    print(f"Error generating query embedding: {e}")
                    flash("Error generating embedding for search query. Falling back to text search.", "warning")
        
        # Fall back to text search if vector search fails or is not selected
        if search_type != "vector" or not vector_results:
            # Ensure text index exists
            ensure_text_index_exists(collection)
            
            # Text search using MongoDB text index or regex
            try:
                # Check if text index exists
                text_search_results = list(collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]))
                
                # If no results, fall back to regex
                if not text_search_results:
                    print(f"No text search results for '{query}', falling back to regex")
                    regex_query = {"$or": [
                        {"Business Name": {"$regex": query, "$options": "i"}},
                        {"Description": {"$regex": query, "$options": "i"}},
                        {"Location": {"$regex": query, "$options": "i"}},
                        {"Type": {"$regex": query, "$options": "i"}},
                        {"Region": {"$regex": query, "$options": "i"}}
                    ]}
                    text_search_results = list(collection.find(regex_query))
                
                # Convert ObjectId to string
                for result in text_search_results:
                    result['_id'] = str(result['_id'])
                
                print(f"Text/regex search found {len(text_search_results)} results for '{query}'")
                return text_search_results
                
            except Exception as e:
                # If text index fails, use regex search
                print(f"Text index search failed: {e}. Using regex search.")
                regex_query = {"$or": [
                    {"Business Name": {"$regex": query, "$options": "i"}},
                    {"Description": {"$regex": query, "$options": "i"}},
                    {"Location": {"$regex": query, "$options": "i"}},
                    {"Type": {"$regex": query, "$options": "i"}},
                    {"Region": {"$regex": query, "$options": "i"}}
                ]}
                results = list(collection.find(regex_query))
                for result in results:
                    result['_id'] = str(result['_id'])
                print(f"Regex search found {len(results)} results for '{query}'")
                return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def find_potential_duplicates():
    collection = get_db_collection()
    try:
        # Find businesses with the same name but different IDs
        pipeline = [
            {"$group": {
                "_id": {"name": "$Business Name"},
                "count": {"$sum": 1},
                "businesses": {"$push": {
                    "_id": "$_id",
                    "Business Name": "$Business Name",
                    "Location": "$Location",
                    "Description": "$Description"
                }}
            }},
            {"$match": {"count": {"$gt": 1}}}
        ]
        
        duplicate_groups = list(collection.aggregate(pipeline))
        
        # Format results for the template
        duplicates = []
        for group in duplicate_groups:
            businesses = group["businesses"]
            for i in range(len(businesses)):
                businesses[i]["_id"] = str(businesses[i]["_id"])
                
            # Create pairs from the businesses in each group
            for i in range(len(businesses)):
                for j in range(i+1, len(businesses)):
                    duplicates.append((businesses[i], businesses[j]))
                    
        return duplicates
    except Exception as e:
        print(f"Error finding duplicates: {e}")
        return []

# Add a new function to check for vector index
def check_vector_index_exists(collection, index_name):
    """Check if vector index exists in MongoDB Atlas collection"""
    try:
        # Try to list search indexes
        indexes = collection.database.command({"listSearchIndexes": collection.name})
        if "indexes" in indexes:
            for index in indexes["indexes"]:
                if index.get("name") == index_name:
                    print(f"Vector index '{index_name}' found")
                    return True
        print(f"Vector index '{index_name}' not found")
        return False
    except pymongo.errors.OperationFailure as e:
        if "command not found" in str(e) or "CommandNotFound" in str(e):
            # This is expected on some MongoDB Atlas tiers that don't support listSearchIndexes
            print(f"Note: Unable to check if index '{index_name}' exists (command not supported). Will attempt to use it.")
            return True  # Assume index exists to prevent recreation attempts
        print(f"Error checking vector index: {e}")
        return False

# Function to create the vector index if it doesn't exist - improved to handle existing index
def ensure_vector_index_exists(collection):
    """Create the vector index if it doesn't exist"""
    index_name = "vector_index"  # Use consistent name with what's in Atlas
    embedding_dim = int(os.getenv("EMBEDDING_DIM", 384))
    
    # Skip index check since listSearchIndexes doesn't work in your Atlas tier
    # Just attempt to create and handle failure gracefully
    try:
        # Create the vector index
        collection.database.command({
            "createSearchIndexes": collection.name,
            "indexes": [{
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "vector_embedding": {
                                "type": "vector",
                                "dimensions": embedding_dim,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            }]
        })
        print(f"Created vector index '{index_name}' with dimension {embedding_dim}")
        return True
    except pymongo.errors.OperationFailure as e:
        if "IndexAlreadyExists" in str(e) or "already defined" in str(e):
            print(f"Vector index '{index_name}' already exists. Using existing index.")
            return True  # Index exists, which is what we want
        print(f"Error creating vector index: {e}")
        return False

# Also create a text index for better text search
def ensure_text_index_exists(collection):
    """Create a text index on relevant fields"""
    try:
        collection.create_index([
            ("Business Name", "text"),
            ("Description", "text"),
            ("Location", "text"),
            ("Type", "text"),
            ("Region", "text")
        ], name="text_search_index")
        print("Text search index created successfully")
        return True
    except Exception as e:
        print(f"Error creating text index: {e}")
        return False

# Add new functions for handling duplicate operations
def get_duplicate_comparison(business_id1, business_id2):
    """Get two businesses to compare for duplication"""
    business1 = get_business_by_id_from_db(business_id1)
    business2 = get_business_by_id_from_db(business_id2)
    return business1, business2

def merge_businesses(business_id1, business_id2, merged_data):
    """Merge two business records and delete the unused one"""
    collection = get_db_collection()
    try:
        # Use ObjectId if the IDs are in that format
        id1 = ObjectId(business_id1) if ObjectId.is_valid(business_id1) else business_id1
        id2 = ObjectId(business_id2) if ObjectId.is_valid(business_id2) else business_id2
        
        # Get both businesses
        business1 = collection.find_one({"_id": id1})
        business2 = collection.find_one({"_id": id2})
        
        # Skip if either business not found
        if not business1 or not business2:
            return False, "One or both businesses not found"
        
        # Create the merged document
        merged_doc = {
            "Business Name": merged_data.get("merged_name", ""),
            "Website": merged_data.get("merged_website", ""),
            "Type": merged_data.get("merged_type", ""),
            "Location": merged_data.get("merged_location", ""),
            "Region": merged_data.get("merged_region", ""),
            "Description": merged_data.get("merged_description", ""),
            "Contact Info": merged_data.get("merged_contact", "")
        }
        
        # Generate combined text for embedding
        cols_for_embedding = ['Business Name', 'Type', 'Description', 'Location', 'Region']
        merged_doc['combined_text_for_embedding'] = ' '.join([str(merged_doc.get(col, '')) for col in cols_for_embedding])
        
        # Preserve vector embedding from one of the records
        if 'vector_embedding' in business1 and business1['vector_embedding']:
            merged_doc['vector_embedding'] = business1['vector_embedding']
        elif 'vector_embedding' in business2 and business2['vector_embedding']:
            merged_doc['vector_embedding'] = business2['vector_embedding']
        else:
            # If no embedding exists, create a placeholder
            merged_doc['vector_embedding'] = [0.0] * int(os.getenv("EMBEDDING_DIM", 384))
        
        # Update the first business with merged data and delete the second one
        collection.update_one({"_id": id1}, {"$set": merged_doc})
        collection.delete_one({"_id": id2})
        
        return True, f"Successfully merged business records. Kept ID {business_id1}, removed ID {business_id2}."
        
    except Exception as e:
        print(f"Error merging businesses: {e}")
        return False, f"Error during merge: {str(e)}"

def ignore_duplicate_pair(business_id1, business_id2):
    """Mark a pair as not duplicates to prevent future flagging"""
    # In a production app, you would save this to a collection of ignored_duplicates
    # For this demo, we'll just return True
    return True

# Add new functions for dashboard data
def get_dashboard_data():
    """Get aggregated data for homepage dashboard"""
    collection = get_db_collection()
    dashboard_data = {}
    
    try:
        # Total number of businesses
        dashboard_data['total_businesses'] = collection.count_documents({})
        
        # Businesses by region - ensure we get regions with count > 0 and limit to top regions
        region_pipeline = [
            {"$match": {"Region": {"$exists": True, "$ne": None, "$ne": ""}}},
            {"$group": {"_id": "$Region", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        region_data = list(collection.aggregate(region_pipeline))
        dashboard_data['regions'] = region_data
        
        # Businesses by type - ensure we get types with count > 0
        type_pipeline = [
            {"$match": {"Type": {"$exists": True, "$ne": None, "$ne": ""}}},
            {"$group": {"_id": "$Type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        type_data = list(collection.aggregate(type_pipeline))
        dashboard_data['types'] = type_data
        
        # Recent activities (last 10 businesses added/modified)
        # In a real system, you'd track creation/modification dates
        # For this demo, we'll just get the most recent documents based on _id
        recent_businesses = list(collection.find().sort([('_id', -1)]).limit(10))
        for business in recent_businesses:
            business['_id'] = str(business['_id'])
        dashboard_data['recent_businesses'] = recent_businesses
        
        # Count businesses with websites vs without
        has_website = collection.count_documents({
            "Website": {"$exists": True, "$ne": None, "$ne": ""}
        })
        dashboard_data['website_stats'] = {
            'has_website': has_website,
            'no_website': dashboard_data['total_businesses'] - has_website
        }
        
        # Businesses by location (top 10)
        location_pipeline = [
            {"$match": {"Location": {"$exists": True, "$ne": None, "$ne": ""}}},
            {"$group": {"_id": "$Location", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        location_data = list(collection.aggregate(location_pipeline))
        dashboard_data['locations'] = location_data
        
        return dashboard_data
    except Exception as e:
        print(f"Error gathering dashboard data: {e}")
        return {
            'total_businesses': 0,
            'regions': [],
            'types': [],
            'recent_businesses': [],
            'website_stats': {'has_website': 0, 'no_website': 0},
            'locations': []
        }

# Add new function to get filter options
def get_filter_options():
    """Extract filter options from database for navigator page"""
    collection = get_db_collection()
    filter_options = {}
    
    try:
        # Get distinct regions and filter out None values
        regions = list(collection.distinct("Region"))
        filter_options['regions'] = sorted([r for r in regions if r is not None])
        
        # Get distinct business types and filter out None values 
        types = list(collection.distinct("Type"))
        filter_options['types'] = sorted([t for t in types if t is not None])
        
        # Get distinct locations and filter out None values
        locations = list(collection.distinct("Location"))
        filter_options['locations'] = sorted([l for l in locations if l is not None])
        
        return filter_options
    except Exception as e:
        print(f"Error getting filter options: {e}")
        return {'regions': [], 'types': [], 'locations': []}

@app.route('/navigator')
@login_required
def business_navigator():
    """Display business navigator with filters and grid view"""
    # Get filter parameters from request
    search_query = request.args.get('query', '')
    selected_region = request.args.get('region', '')
    selected_type = request.args.get('type', '')
    selected_location = request.args.get('location', '')
    page = request.args.get('page', 1, type=int)
    per_page = 100  # Show more items in grid view
    
    collection = get_db_collection()
    try:
        # Build filter query
        filter_query = {}
        
        if search_query:
            # Add text search if query provided
            filter_query['$or'] = [
                {"Business Name": {"$regex": search_query, "$options": "i"}},
                {"Description": {"$regex": search_query, "$options": "i"}},
                {"Type": {"$regex": search_query, "$options": "i"}}
            ]
        
        if selected_region:
            filter_query['Region'] = selected_region
            
        if selected_type:
            filter_query['Type'] = selected_type
            
        if selected_location:
            filter_query['Location'] = selected_location
        
        # Get total count with filters
        total_count = collection.count_documents(filter_query)
        
        # Calculate pagination
        skip_count = (page - 1) * per_page
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        
        # Get businesses with filters and pagination
        businesses = list(collection.find(filter_query).skip(skip_count).limit(per_page))
        
        # Convert ObjectId to string
        for business in businesses:
            business['_id'] = str(business['_id'])
        
        # Get all filter options for sidebar
        filter_options = get_filter_options()
        
        return render_template(
            'business_navigator.html',
            businesses=businesses,
            filter_options=filter_options,
            selected_region=selected_region,
            selected_type=selected_type,
            selected_location=selected_location,
            search_query=search_query,
            page=page,
            total_pages=total_pages,
            total_count=total_count
        )
    except Exception as e:
        flash(f"Error loading business navigator: {e}", "error")
        return render_template('business_navigator.html', businesses=[], filter_options={}, page=1, total_pages=1)

# --- Flask Routes ---
@app.route('/')
@login_required
def index():
    """Display dashboard homepage"""
    try:
        dashboard_data = get_dashboard_data()
        return render_template('dashboard.html', data=dashboard_data)
    except Exception as e:
        flash(f"Error loading dashboard: {e}", "error")
        return render_template('dashboard.html', data={})

@app.route('/businesses')
@login_required
def business_list():
    """Display list of businesses (moved from the original index route)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 100
        businesses, total_businesses = get_all_businesses_from_db(page, per_page)
        total_pages = (total_businesses + per_page - 1) // per_page
        return render_template('business_list.html', businesses=businesses, page=page, total_pages=total_pages, per_page=per_page)
    except Exception as e:
        flash(f"Error loading businesses: {e}", "error")
        return render_template('business_list.html', businesses=[], page=1, total_pages=1, per_page=100)

@app.route('/business/<business_id>')
@login_required
def business_detail(business_id):
    business = get_business_by_id_from_db(business_id)
    if business:
        return render_template('business_detail.html', business=business)
    flash(f"Business with ID {business_id} not found.", "error")
    return redirect(url_for('index'))

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_business():
    if request.method == 'POST':
        # Basic data collection, no validation for brevity
        new_data = {
            "Business Name": request.form.get("business_name"),
            "Website": request.form.get("website"),
            "Location": request.form.get("location"),
            "Type": request.form.get("type"),
            "Description": request.form.get("description"),
            "Contact Info": request.form.get("contact_info"),
            "Region": request.form.get("region"),
        }
        # Add business to database
        business_id = add_business_to_db(new_data)
        if business_id:
            flash(f"Business '{new_data['Business Name']}' added successfully with ID {business_id}!", "success")
            return redirect(url_for('business_detail', business_id=business_id))
        else:
            flash("Failed to add business.", "error")
            return render_template('add_business.html', business=new_data)
    return render_template('add_business.html')

@app.route('/edit/<business_id>', methods=['GET', 'POST'])
@login_required
def edit_business(business_id):
    business = get_business_by_id_from_db(business_id)
    if not business:
        flash(f"Business with ID {business_id} not found.", "error")
        return redirect(url_for('index'))

    if request.method == 'POST':
        updated_data = {
            "Business Name": request.form.get("business_name"),
            "Website": request.form.get("website"),
            "Location": request.form.get("location"),
            "Type": request.form.get("type"),
            "Description": request.form.get("description"),
            "Contact Info": request.form.get("contact_info"),
            "Region": request.form.get("region"),
        }
        if update_business_in_db(business_id, updated_data):
            flash(f"Business '{updated_data['Business Name']}' updated successfully!", "success")
        else:
            flash("Failed to update business.", "error")
        return redirect(url_for('business_detail', business_id=business_id))
    return render_template('edit_business.html', business=business)

@app.route('/delete/<business_id>', methods=['POST']) # Use POST for destructive actions
@login_required
def delete_business_action(business_id):
    business = get_business_by_id_from_db(business_id) # Get name for flash message
    if business and delete_business_from_db(business_id):
        flash(f"Business '{business.get('Business Name', 'N/A')}' deleted successfully!", "success")
    else:
        flash(f"Failed to delete business with ID {business_id}.", "error")
    return redirect(url_for('index'))

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form.get('query', '')
        search_type = request.form.get('search_type', 'text') # 'text' or 'vector'

        results = []
        if query:
            results = search_businesses_in_db(query, search_type)
            
            if not results:
                flash(f"No results found for '{query}'.", "info")
        else:
            flash("Please enter a search query.", "warning")
            
        return render_template('search_results.html', query=query, results=results, search_type=search_type)
    return render_template('search_form.html') # A page with the search form

@app.route('/duplicates')
@login_required
def find_duplicates():
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 100  # Show 100 duplicates at a time
    
    potential_duplicates = find_potential_duplicates()
    
    if not potential_duplicates:
        flash("No duplicate businesses found based on name matching.", "info")
        return render_template('duplicates.html', duplicates=[], page=page, total_pages=1, total_count=0)
    
    # Calculate pagination values
    total_count = len(potential_duplicates)
    total_pages = (total_count + per_page - 1) // per_page  # Ceiling division
    
    # Slice the duplicates list for the current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_duplicates = potential_duplicates[start_idx:end_idx]
    
    return render_template(
        'duplicates.html', 
        duplicates=paginated_duplicates,
        page=page,
        total_pages=total_pages,
        total_count=total_count
    )

@app.route('/enrich')
@login_required
def enrich_data_page():
    try:
        # Get a sample of businesses to enrich
        collection = get_db_collection()
        businesses = list(collection.find({}).limit(5))
        for business in businesses:
            business['_id'] = str(business['_id'])
        
        flash("Data enrichment with LLMs is conceptual. This page is a placeholder.", "info")
        return render_template('enrich.html', businesses=businesses)
    except Exception as e:
        flash(f"Error loading businesses: {e}", "error")
        return render_template('enrich.html', businesses=[])

# Add routes for handling duplicate comparison and merging
@app.route('/duplicates/compare/<business_id1>/<business_id2>')
@login_required
def compare_duplicates(business_id1, business_id2):
    """Compare two potential duplicate businesses side by side"""
    business1, business2 = get_duplicate_comparison(business_id1, business_id2)
    
    if not business1 or not business2:
        flash("One or both of the businesses for comparison could not be found.", "error")
        return redirect(url_for('find_duplicates'))
    
    return render_template('compare_duplicates.html', business1=business1, business2=business2)

@app.route('/duplicates/merge', methods=['POST'])
@login_required
def merge_duplicates():
    """Process the merge of two duplicate businesses"""
    business_id1 = request.form.get('business_id1')
    business_id2 = request.form.get('business_id2')
    
    if not business_id1 or not business_id2:
        flash("Missing business IDs for merge operation.", "error")
        return redirect(url_for('find_duplicates'))
    
    merged_data = {
        "merged_name": request.form.get('merged_name', ''),
        "merged_website": request.form.get('merged_website', ''),
        "merged_type": request.form.get('merged_type', ''),
        "merged_location": request.form.get('merged_location', ''),
        "merged_region": request.form.get('merged_region', ''),
        "merged_description": request.form.get('merged_description', ''),
        "merged_contact": request.form.get('merged_contact', '')
    }
    
    success, message = merge_businesses(business_id1, business_id2, merged_data)
    
    if success:
        flash(message, "success")
    else:
        flash(message, "error")
    
    return redirect(url_for('find_duplicates'))

@app.route('/duplicates/ignore/<business_id1>/<business_id2>')
@login_required
def ignore_duplicates(business_id1, business_id2):
    """Mark a pair of businesses as 'not duplicates'"""
    if ignore_duplicate_pair(business_id1, business_id2):
        flash(f"Businesses marked as not duplicates. They won't be suggested again.", "info")
    else:
        flash("Failed to ignore duplicate pair.", "error")
    
    return redirect(url_for('find_duplicates'))

# Add CSV export functionality
import csv
from io import StringIO
from datetime import datetime

def generate_csv(collection_name):
    """Generate a CSV from a specific collection"""
    collection = get_db_collection(collection_name)
    if not collection:
        return None, f"Collection '{collection_name}' not found"
    
    try:
        # Get all documents from the collection
        documents = list(collection.find())
        if not documents:
            return None, "No data to export"
        
        # Use the first document to determine field names
        fieldnames = documents[0].keys()
        
        # Create CSV output
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Convert ObjectId to string for each document
        for doc in documents:
            doc['_id'] = str(doc['_id'])
            writer.writerow(doc)
            
        output.seek(0)
        return output, None
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return None, f"Error generating CSV: {str(e)}"

# Add routes for waitlist and newsletter
    if request.method == 'POST':
        try:
            # Extract form data
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            company_name = request.form.get('company_name', '')
            business_type = request.form.get('business_type', '')
            
            # Create a document
            account_entry = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "company_name": company_name,
                "business_type": business_type,
                "date_created": datetime.now(),
                "source": "account_form"
            }
            
            # Save to MongoDB
            collection = get_db_collection("newsletter")
            if collection:
                # Check if email already exists
                existing_user = collection.find_one({"email": email})
                if existing_user:
                    flash("An account with this email already exists", "warning")
                else:
                    result = collection.insert_one(account_entry)
                    if result.inserted_id:
                        success_message = "Your account has been created successfully!"
            else:
                flash("An error occurred while processing your request", "error")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    
    return render_template('newsletter.html', success_message=success_message)

@app.route('/export/<collection_name>')
def export_data(collection_name):
    """Export a specific collection as CSV"""
    # Check authentication (in a real app, add proper auth check)
    if not collection_name or collection_name not in ["waitlist", "newsletter"]:
        flash("Invalid collection name", "error")
        return redirect(url_for('index'))
    
    csv_data, error = generate_csv(collection_name)
    if error:
        flash(error, "error")
        return redirect(url_for('index'))
    
    # Create a response with the CSV data
    response = app.response_class(
        response=csv_data,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename={collection_name}-{datetime.now().strftime("%Y%m%d")}.csv'
        }
    )
    
    return response

# Update routes for direct data entry without templates
@app.route('/admin/waitlist', methods=['POST'])
def add_to_waitlist():
    """Add an entry to the waitlist directly from admin page"""
    try:
        # Extract form data
        email = request.form.get('email')
        name = request.form.get('name')
        role = request.form.get('role')
        interests = request.form.getlist('interests')  # Get multiple checkbox values
        subscribe_newsletter = 'subscribe_newsletter' in request.form
        
        # Create a document
        waitlist_entry = {
            "email": email,
            "name": name,
            "role": role,
            "interests": interests,
            "subscribe_newsletter": subscribe_newsletter,
            "date_added": datetime.now(),
            "source": "admin_form"
        }
        
        # Save to MongoDB
        collection = get_db_collection("waitlist")
        if collection:
            result = collection.insert_one(waitlist_entry)
            if result.inserted_id:
                flash("Successfully added entry to waitlist", "success")
                # Add to newsletter list if selected
                if subscribe_newsletter:
                    newsletter_collection = get_db_collection("newsletter")
                    if newsletter_collection:
                        # Check if email already exists in newsletter
                        if not newsletter_collection.find_one({"email": email}):
                            newsletter_collection.insert_one({
                                "email": email,
                                "name": name,
                                "date_added": datetime.now(),
                                "source": "waitlist_form"
                            })
        else:
            flash("Failed to add entry to waitlist", "error")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    
    return redirect(url_for('admin'))

@app.route('/admin/newsletter', methods=['POST'])
def add_to_newsletter():
    """Add a subscriber to the newsletter directly from admin page"""
    try:
        # Extract form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        company_name = request.form.get('company_name', '')
        business_type = request.form.get('business_type', '')
        
        # Create a document
        account_entry = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "company_name": company_name,
            "business_type": business_type,
            "date_created": datetime.now(),
            "source": "admin_form"
        }
        
        # Save to MongoDB
        collection = get_db_collection("newsletter")
        if collection:
            # Check if email already exists
            existing_user = collection.find_one({"email": email})
            if existing_user:
                flash("An account with this email already exists", "warning")
            else:
                result = collection.insert_one(account_entry)
                if result.inserted_id:
                    flash("Successfully added subscriber to newsletter", "success")
        else:
            flash("Failed to add subscriber to newsletter", "error")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    
    return redirect(url_for('admin'))

@app.route('/upload-csv/<collection_name>', methods=['POST'])
def upload_csv(collection_name):
    """Handle CSV file upload for bulk data import"""
    if collection_name not in ['waitlist', 'newsletter']:
        flash("Invalid collection name", "error")
        return redirect(url_for('admin'))
    
    try:
        if 'csv_file' not in request.files:
            flash("No file part", "error")
            return redirect(url_for('admin'))
        
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            flash("No selected file", "error")
            return redirect(url_for('admin'))
        
        if csv_file and csv_file.filename.endswith('.csv'):
            # Read CSV file
            csv_data = []
            stream = StringIO(csv_file.stream.read().decode("utf-8"))
            reader = csv.DictReader(stream)
            
            for row in reader:
                # Add timestamp
                row['date_added'] = datetime.now()
                row['source'] = 'csv_upload'
                csv_data.append(row)
            
            if csv_data:
                # Insert into MongoDB
                collection = get_db_collection(collection_name)
                if collection:
                    result = collection.insert_many(csv_data)
                    flash(f"Successfully imported {len(result.inserted_ids)} records into {collection_name}", "success")
                else:
                    flash(f"Failed to connect to {collection_name} collection", "error")
            else:
                flash("No data found in CSV file", "warning")
        else:
            flash("File must be a CSV", "error")
    
    except Exception as e:
        flash(f"Error processing CSV file: {str(e)}", "error")
    
    return redirect(url_for('admin'))

# Update admin route to include recent entries
@app.route('/admin')
def admin():
    """Admin dashboard for data export and management"""
    # In a real app, add authentication checks here
    
    try:
        # Get counts for each collection
        waitlist_collection = get_db_collection("waitlist")
        newsletter_collection = get_db_collection("newsletter")
        
        # Fix: Change bool check to explicit None comparison
        waitlist_count = waitlist_collection.count_documents({}) if waitlist_collection is not None else 0
        newsletter_count = newsletter_collection.count_documents({}) if newsletter_collection is not None else 0
        
        return render_template(
            'admin.html',
            waitlist_count=waitlist_count,
            newsletter_count=newsletter_count
        )
    except Exception as e:
        flash(f"Error loading admin dashboard: {e}", "error")
        return render_template('admin.html', waitlist_count=0, newsletter_count=0)

# Also add a link in the sidebar or navigation menu
# You could update the base.html template, or add links in a new admin page

# --- HTML Templates management ---
def create_dummy_templates():
    """Create template files only if they don't already exist"""
    if not os.path.exists("templates"):
        os.makedirs("templates")

    # Define a helper function to create a file only if it doesn't exist
    def create_file_if_not_exists(filename, content):
        filepath = os.path.join("templates", filename)
        if not os.path.exists(filepath):
            print(f"Creating template file: {filename}")
            with open(filepath, "w") as f:
                f.write(content)
        else:
            print(f"Template file already exists: {filename}")

    # Define template content and create only missing files
    base_html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
        <title>{% block title %}Tea Knowledge Graph{% endblock %}</title>
        <style>
            body { padding-top: 70px; }
            .table-responsive { margin-top: 20px; }
            .flash-messages .alert { margin-bottom: 10px; }
            .nav-item .nav-link { padding: 0.5rem 1rem; }
            .nav-item .nav-link i { margin-right: 5px; }
        </style>
        {% block head_extra %}{% endblock %}
    </head>
    <body>
        <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
            <a class="navbar-brand" href="{{ url_for('index') }}"><i class="fas fa-leaf mr-2"></i>Chaiport Database</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarsExampleDefault" aria-controls="navbarsExampleDefault" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarsExampleDefault">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('index') }}"><i class="fas fa-chart-line"></i> Dashboard</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('business_navigator') }}"><i class="fas fa-compass"></i> Business Navigator</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('business_list') }}"><i class="fas fa-list"></i> All Businesses</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('add_business') }}"><i class="fas fa-plus-circle"></i> Add Business</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('search') }}"><i class="fas fa-search"></i> Search</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('find_duplicates') }}"><i class="fas fa-clone"></i> Find Duplicates</a></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('enrich_data_page') }}"><i class="fas fa-magic"></i> Enrich Data</a></li>
                </ul>
                <form class="form-inline my-2 my-lg-0" action="{{ url_for('business_navigator') }}" method="GET">
                    <input class="form-control mr-sm-2" type="search" name="query" placeholder="Quick search..." aria-label="Search">
                    <button class="btn btn-outline-success my-2 my-sm-0" type="submit">Search</button>
                </form>
            </div>
        </nav>
        <div class="container">
            <div class="flash-messages">
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                    <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                        <span aria-hidden="true">×</span>
                        </button>
                    </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            </div>
            {% block content %}{% endblock %}
            
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        // Highlight active nav item
        $(document).ready(function() {
            var currentUrl = window.location.pathname;
            $('.navbar-nav .nav-link').each(function() {
                var linkUrl = $(this).attr('href');
                if (linkUrl !== '/' && currentUrl.indexOf(linkUrl) !== -1) {
                    $(this).addClass('active');
                } else if (linkUrl === '/' && currentUrl === '/') {
                    $(this).addClass('active');
                }
            });
        });
        </script>
        {% block scripts %}{% endblock %}
    </body>
    </html>
    """
    create_file_if_not_exists("base.html", base_html)

    index_html = """
    {% extends "base.html" %}
    {% block title %}All Tea Businesses{% endblock %}
    {% block content %}
        <h1>Tea Businesses</h1>
        <div class="table-responsive">
            <table class="table table-striped table-sm">
                <thead>
                    <tr>
                        <th>Name</th><th>Type</th><th>Location</th><th>Region</th><th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                {% for business in businesses %}
                    <tr>
                        <td><a href="{{ url_for('business_detail', business_id=business['_id']|string) }}">{{ business['Business Name'] }}</a></td>
                        <td>{{ business['Type'] }}</td>
                        <td>{{ business['Location'] }}</td>
                        <td>{{ business['Region'] }}</td>
                        <td>
                            <a href="{{ url_for('edit_business', business_id=business['_id']|string) }}" class="btn btn-sm btn-outline-primary">Edit</a>
                            <form action="{{ url_for('delete_business_action', business_id=business['_id']|string) }}" method="POST" style="display:inline;">
                                <button type="submit" class="btn btn-sm btn-outline-danger" onclick="return confirm('Are you sure you want to delete this business?');">Delete</button>
                            </form>
                        </td>
                    </tr>
                {% else %}
                    <tr><td colspan="5">No businesses found.</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% if total_pages > 1 %}
        <nav aria-label="Page navigation">
            <ul class="pagination">
                {% if page > 1 %}
                <li class="page-item"><a class="page-link" href="{{ url_for('index', page=page-1) }}">Previous</a></li>
                {% endif %}
                {% for p in range(1, total_pages + 1) %}
                <li class="page-item {% if p == page %}active{% endif %}"><a class="page-link" href="{{ url_for('index', page=p) }}">{{ p }}</a></li>
                {% endfor %}
                {% if page < total_pages %}
                <li class="page-item"><a class="page-link" href="{{ url_for('index', page=page+1) }}">Next</a></li>
                {% endif %}
            </ul>
        </nav>
        {% endif %}
    {% endblock %}
    """
    create_file_if_not_exists("index.html", index_html)

    business_detail_html = """
    {% extends "base.html" %}
    {% block title %}{{ business['Business Name'] }}{% endblock %}
    {% block content %}
        <h1>{{ business['Business Name'] }}</h1>
        <p><strong>ID:</strong> {{ business['_id'] }}</p>
        <p><strong>Type:</strong> {{ business['Type'] }}</p>
        <p><strong>Location:</strong> {{ business['Location'] }}</p>
        <p><strong>Region:</strong> {{ business['Region'] }}</p>
        <p><strong>Website:</strong> <a href="{{ business['Website'] }}" target="_blank">{{ business['Website'] }}</a></p>
        <p><strong>Description:</strong> {{ business['Description'] }}</p>
        <p><strong>Contact Info:</strong> {{ business['Contact Info'] }}</p>
        <p><strong>Combined Text for Embedding (Sample):</strong> {{ business['combined_text_for_embedding'][:200] }}...</p>
        <p><strong>Vector Embedding (Sample):</strong> {{ business['vector_embedding'][:5] if business['vector_embedding'] }}...</p>
        <a href="{{ url_for('edit_business', business_id=business['_id']|string) }}" class="btn btn-primary">Edit</a>
        <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to List</a>
    {% endblock %}
    """
    create_file_if_not_exists("business_detail.html", business_detail_html)

    # Keep defining other templates...
    form_snippet_fields = """
        <div class="form-group">
            <label for="business_name">Business Name</label>
            <input type="text" class="form-control" id="business_name" name="business_name" value="{{ business['Business Name'] if business else '' }}" required>
        </div>
        <div class="form-group">
            <label for="website">Website</label>
            <input type="url" class="form-control" id="website" name="website" value="{{ business['Website'] if business else '' }}">
        </div>
        <div class="form-group">
            <label for="location">Location</label>
            <input type="text" class="form-control" id="location" name="location" value="{{ business['Location'] if business else '' }}">
        </div>
        <div class="form-group">
            <label for="type">Type</label>
            <input type="text" class="form-control" id="type" name="type" value="{{ business['Type'] if business else '' }}">
        </div>
        <div class="form-group">
            <label for="description">Description</label>
            <textarea class="form-control" id="description" name="description" rows="3">{{ business['Description'] if business else '' }}</textarea>
        </div>
        <div class="form-group">
            <label for="contact_info">Contact Info</label>
            <input type="text" class="form-control" id="contact_info" name="contact_info" value="{{ business['Contact Info'] if business else '' }}">
        </div>
        <div class="form-group">
            <label for="region">Region</label>
            <input type="text" class="form-control" id="region" name="region" value="{{ business['Region'] if business else '' }}">
        </div>
    """
    
    add_business_html = f"""
    {{% extends "base.html" %}}
    {{% block title %}}Add New Business{{% endblock %}}
    {{% block content %}}
        <h1>Add New Tea Business</h1>
        <form method="POST">
            {form_snippet_fields}
            <button type="submit" class="btn btn-success">Add Business</button>
        </form>
    {{% endblock %}}
    """
    create_file_if_not_exists("add_business.html", add_business_html)

    edit_business_html = f"""
    {{% extends "base.html" %}}
    {{% block title %}}Edit {{ business['Business Name'] }}{{% endblock %}}
    {{% block content %}}
        <h1>Edit: {{ business['Business Name'] }}</h1>
        <form method="POST">
            {form_snippet_fields}
            <button type="submit" class="btn btn-primary">Save Changes</button>
            <a href="{{ url_for('business_detail', business_id=business['_id']|string) }}" class="btn btn-secondary">Cancel</a>
        </form>
    {{% endblock %}}
    """
    create_file_if_not_exists("edit_business.html", edit_business_html)

    search_form_html = """
    {% extends "base.html" %}
    {% block title %}Search Businesses{% endblock %}
    {% block content %}
        <h1>Search Tea Businesses</h1>
        <form method="POST" action="{{ url_for('search') }}">
            <div class="form-group">
                <label for="query">Search Query</label>
                <input type="text" class="form-control" id="query" name="query" placeholder="Enter name, description, location, etc." required>
            </div>
            <div class="form-group">
                <label for="search_type">Search Type</label>
                <select class="form-control" id="search_type" name="search_type">
                    <option value="text">Text Search</option>
                    <option value="vector">Vector Similarity (Conceptual)</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary">Search</button>
        </form>
    {% endblock %}
    """
    create_file_if_not_exists("search_form.html", search_form_html)

    search_results_html = """
    {% extends "base.html" %}
    {% block title %}Search Results for "{{ query }}"{% endblock %}
    {% block content %}
        <h1>Search Results for "{{ query }}" <small class="text-muted">({{ search_type }} search)</small></h1>
        {% if results %}
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead><tr><th>Name</th><th>Type</th><th>Location</th><th>Description</th>{% if results[0].score %}<th>Score</th>{% endif %}</tr></thead>
                    <tbody>
                    {% for business in results %}
                        <tr>
                            <td><a href="{{ url_for('business_detail', business_id=business['_id']|string) }}">{{ business['Business Name'] }}</a></td>
                            <td>{{ business['Type'] }}</td>
                            <td>{{ business['Location'] }}</td>
                            <td>{{ business['Description'][:100] }}...</td>
                            {% if business.score %}<td>{{ "%.4f"|format(business.score) }}</td>{% endif %}
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p>No results found matching your query.</p>
        {% endif %}
        <a href="{{ url_for('search') }}" class="btn btn-secondary mt-3">New Search</a>
    {% endblock %}
    """
    create_file_if_not_exists("search_results.html", search_results_html)

    duplicates_html = """
    {% extends "base.html" %}
    {% block title %}Potential Duplicates{% endblock %}
    {% block content %}
        <h1>Potential Duplicates</h1>
        <p class="text-muted">(Based on simple name match for this demo. Real duplicate detection would use vector similarity or more advanced heuristics.)</p>
        {% if duplicates %}
            {% for b1, b2 in duplicates %}
                <div class="card mb-3">
                    <div class="card-header">Potential Duplicate Pair</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5><a href="{{ url_for('business_detail', business_id=b1['_id']|string) }}">{{ b1['Business Name'] }} (ID: {{ b1['_id'] }})</a></h5>
                                <p><strong>Location:</strong> {{ b1['Location'] }}</p>
                                <p><strong>Description:</strong> {{ b1['Description'][:100] }}...</p>
                            </div>
                            <div class="col-md-6">
                                <h5><a href="{{ url_for('business_detail', business_id=b2['_id']|string) }}">{{ b2['Business Name'] }} (ID: {{ b2['_id'] }})</a></h5>
                                <p><strong>Location:</strong> {{ b2['Location'] }}</p>
                                <p><strong>Description:</strong> {{ b2['Description'][:100] }}...</p>
                            </div>
                        </div>
                         <!-- Add merge/ignore buttons here in a real app -->
                    </div>
                </div>
            {% endfor %}
        {% else %}
            <p>No obvious duplicates found with the current simple check.</p>
        {% endif %}
    {% endblock %}
    """
    create_file_if_not_exists("duplicates.html", duplicates_html)

    enrich_html = """
    {% extends "base.html" %}
    {% block title %}Enrich Data{% endblock %}
    {% block content %}
        <h1>Enrich Tea Business Data (Conceptual)</h1>
        <p>Select a business to (conceptually) enrich its data using an LLM.</p>
        <ul class="list-group">
        {% for business in businesses %}
            <li class="list-group-item">
                {{ business['Business Name'] }} - {{ business['Location'] }}
                <button class="btn btn-sm btn-info float-right" disabled>Enrich (Conceptual)</button>
            </li>
        {% else %}
            <li class="list-group-item">No businesses to display for enrichment.</li>
        {% endfor %}
        </ul>
        <div class="mt-3">
            <p class="text-muted"><strong>How this would work:</strong></p>
            <ol>
                <li>User selects a business.</li>
                <li>The system takes key info (e.g., name, description).</li>
                <li>This info is sent to an LLM (e.g., "Provide a more detailed description for 'XYZ Tea Estate' focusing on its unique offerings and history.").</li>
                <li>The LLM's response is presented to the user for review.</li>
                <li>User can edit and then choose to update the business record with the enriched information.</li>
            </ol>
        </div>
    {% endblock %}
    """
    create_file_if_not_exists("enrich.html", enrich_html)

# create_dummy_templates() # Create template files

if __name__ == '__main__':
    # Verify MongoDB connection before starting
    try:
        client = get_mongo_client()
        db_name = os.getenv("DB_NAME", "tea_knowledge_graph")
        collection_name = os.getenv("COLLECTION_NAME", "tea_businesses")
        collection = client[db_name][collection_name]
        count = collection.count_documents({})
        print(f"Successfully connected to MongoDB. Collection '{collection_name}' has {count} documents.")
        
        # Check and create indexes at startup
        ensure_text_index_exists(collection)
        if SENTENCE_TRANSFORMERS_AVAILABLE and load_embedding_model():
            print("Vector search capability is available.")
            ensure_vector_index_exists(collection)
        else:
            print("Vector search not available. Install sentence-transformers to enable this feature.")
    except Exception as e:
        print(f"Warning: MongoDB connection failed: {e}")
        print("The app will start, but database features won't work until connection is fixed.")
    
    # Start Flask app
    port = int(os.environ.get('PORT', 8080))
    gunicorn -w 1 -b 0.0.0.0:$PORT flask_app:app