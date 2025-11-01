import joblib
import pandas as pd
import traceback

print('Loading model...')
model = joblib.load('model.pkl')
print('Model type:', type(model))

for attr in ['n_features_in_', 'feature_names_in_', 'classes_', 'get_params']:
    print(attr, '->', getattr(model, attr, 'MISSING'))

print('Has predict:', hasattr(model, 'predict'))

input_df = pd.DataFrame({
    'area_sqft': [1000],
    'bedrooms': [2],
    'baths': [1],
    'city': ['Lahore'],
    'property_type': ['House'],
    'purpose': ['For Sale'],
    'province_name': ['Punjab']
})

print('\nAttempting model.predict on raw DataFrame (may raise)...')
try:
    out = model.predict(input_df)
    print('Predict output:', out)
except Exception as e:
    print('Predict raised:')
    traceback.print_exc()
