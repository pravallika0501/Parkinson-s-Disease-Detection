import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\admin\Desktop\Parkinson's Disease Detection\parkinsons.data.csv")

# Select only 8 relevant features
features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:RAP',
            'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']

X = df[features]
y = df["status"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
