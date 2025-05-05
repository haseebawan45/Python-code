import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
cities = ['Karachi', 'Lahore', 'Islamabad', 'Quetta', 'Peshawar']
data = pd.DataFrame({
    'City': np.random.choice(cities, 500),
    'Temperature': np.random.uniform(10, 45, 500), 
    'Humidity': np.random.uniform(20, 80, 500),     
    'WindSpeed': np.random.uniform(0, 15, 500),     
    'Precipitation': np.random.uniform(0, 20, 500), 
    'ExpectedTemp': np.random.uniform(15, 40, 500)  
})

X = data[['Temperature', 'Humidity', 'WindSpeed', 'Precipitation']]
y = data['ExpectedTemp']

X = pd.get_dummies(data[['City']].join(X), columns=['City'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

city_means = data.groupby('City')[['Temperature', 'Humidity', 'WindSpeed', 'Precipitation']].mean()
city_means = pd.get_dummies(city_means, columns=['City'], drop_first=True)
predicted_temps = model.predict(city_means)

print("\nExpected Future Temperatures by City:")
for city, temp in zip(cities, predicted_temps):
    print(f"{city}: {temp:.2f} °C")
