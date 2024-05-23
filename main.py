import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Завантаження даних з файлу Excel
data = pd.read_excel('housing_data.xlsx')

print(data)
# One-hot encoding для категоріальної змінної 'location'
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Відокремлення ознак та цільової змінної
X = data.drop('price', axis=1)
y = data['price']

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Додавання колонки "location_Rural" у вхідні дані
X_train['location_Rural'] = 0  # Додаємо колонку і заповнюємо її значеннями 0
X_test['location_Rural'] = 0

# Вибір моделі
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = model.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.1f}')
print(f'R2 Score: {r2:.3f}')

# Збереження моделі
joblib.dump(model, 'house_price_model.pkl')


# Завантаження моделі
loaded_model = joblib.load('house_price_model.pkl')


# Функція для введення даних користувачем
def get_user_input():
    square_feet = float(input("Enter square feet: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    year_built = int(input("Enter year built: "))
    location = input("Enter location (Suburban/Urban/Rural): ")

    location_suburban = 1 if location.lower() == 'suburban' else 0
    location_urban = 1 if location.lower() == 'urban' else 0
    location_rural = 1 if location.lower() == 'rural' else 0

    user_data = {
        'square_feet': [square_feet],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'year_built': [year_built],
        'location_Suburban': [location_suburban],
        'location_Urban': [location_urban],
        'location_Rural': [location_rural]
    }

    return pd.DataFrame(user_data)


# Отримання введених даних
new_data = get_user_input()

# Прогнозування для введених даних
price_prediction = loaded_model.predict(new_data)
print(f'Predicted price: {price_prediction[0]:.2f}')

