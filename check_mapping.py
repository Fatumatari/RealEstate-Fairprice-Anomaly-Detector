import pandas as pd
import pickle

# Read the data
df = pd.read_csv('data/processed/cleaned_listings.csv')

# Create state-locality mapping
state_locality = df[['state', 'locality']].drop_duplicates()
state_locality_dict = state_locality.groupby('state')['locality'].apply(sorted).apply(list).to_dict()

print('Sample mapping (first 3 states):')
for i, (state, localities) in enumerate(state_locality_dict.items()):
    if i < 3:
        print(f'{state}: {localities[:5]} ... ({len(localities)} total)')

# Save it
with open('models/state_locality_mapping.pkl', 'wb') as f:
    pickle.dump(state_locality_dict, f)
    
print('\nMapping saved to models/state_locality_mapping.pkl')
