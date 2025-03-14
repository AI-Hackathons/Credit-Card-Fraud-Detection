import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Faker and set random seed
fake = Faker()
np.random.seed(42)

def generate_transaction_data(num_records=10000, fraud_percentage=0.05, start_date='-30d'):
    """
    Generate synthetic mobile money transaction data with fraud patterns.
    
    Parameters:
    - num_records: Number of transactions to generate
    - fraud_percentage: Proportion of fraudulent transactions (0-1)
    - start_date: Starting date for transactions (e.g., '-30d' for 30 days ago)
    """
    try:
        logger.info(f"Generating {num_records} transactions with {fraud_percentage*100}% fraud rate")
        
        data = []
        fraud_count = int(num_records * fraud_percentage)
        
        # Enhanced location data with country codes
        locations = {
            'KE': ['Nairobi', 'Mombasa', 'Kisumu'],
            'UG': ['Kampala', 'Entebbe'],
            'TZ': ['Dar es Salaam', 'Arusha'],
            'RW': ['Kigali'],
            'BI': ['Bujumbura']
        }
        
        # Customer base for repeat transactions
        customer_base = [fake.uuid4()[:8] for _ in range(int(num_records/10))]
        
        for i in range(num_records):
            is_fraud = int(i < fraud_count)
            timestamp = fake.date_time_between(start_date=start_date, end_date='now')
            customer_id = np.random.choice(customer_base)
            
            # Base transaction characteristics
            amount = (abs(np.random.normal(500, 200)) if is_fraud 
                     else max(1, np.random.normal(100, 50)))
            
            country = np.random.choice(list(locations.keys()))
            location = f"{country}-{np.random.choice(locations[country])}"
            
            transaction = {
                'transaction_id': fake.uuid4(),
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'customer_id': customer_id,
                'customer_age': np.random.randint(18, 70),
                'customer_gender': np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04]),
                'location': location,
                'transaction_type': np.random.choice(
                    ['P2P', 'Cashout', 'Pay Bill', 'Airtime', 'Deposit'],
                    p=[0.35, 0.25, 0.20, 0.15, 0.05]
                ),
                'account_balance_before': round(np.random.uniform(100, 10000), 2),
                'device_type': np.random.choice(
                    ['Mobile App', 'USSD', 'Agent'],
                    p=[0.6, 0.3, 0.1]
                ),
                'network_provider': np.random.choice(
                    ['Safaricom', 'MTN', 'Airtel'],
                    p=[0.5, 0.3, 0.2]
                ),
                'failed_attempts': np.random.poisson(2 if is_fraud else 0.1),
                'ip_address': fake.ipv4() if np.random.rand() > 0.3 else None,
                'is_fraud': is_fraud
            }
            
            # Enhanced fraud patterns
            if is_fraud:
                if np.random.rand() > 0.7:
                    transaction['location'] = 'XX-Foreign'
                if np.random.rand() > 0.8:
                    transaction['timestamp'] = timestamp.replace(
                        hour=np.random.randint(0, 5),
                        minute=np.random.randint(0, 59)
                    )
                if np.random.rand() > 0.6:
                    transaction['failed_attempts'] = np.random.randint(3, 10)
            
            # Calculate balance after transaction
            transaction['account_balance_after'] = max(
                0, transaction['account_balance_before'] - transaction['amount']
            )
            
            data.append(transaction)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add derived features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['amount_to_balance_ratio'] = df['amount'] / df['account_balance_before']
        df['country_code'] = df['location'].str.split('-').str[0]
        
        # Validate data
        assert df['amount'].min() > 0, "Negative amounts detected"
        assert df['account_balance_after'].min() >= 0, "Negative balances detected"
        
        logger.info("Data generation completed successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        raise

def analyze_dataset(df):
    """Basic analysis of the generated dataset"""
    print("\nDataset Summary:")
    print("-" * 50)
    print(df.describe())
    print("\nFraud Statistics:")
    print("-" * 50)
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Average fraud amount: ${df[df['is_fraud'] == 1]['amount'].mean():.2f}")
    print(f"Average legit amount: ${df[df['is_fraud'] == 0]['amount'].mean():.2f}")
    print("\nMissing Values:")
    print("-" * 50)
    print(df.isnull().sum())

if __name__ == "__main__":
    # Generate dataset
    transaction_df = generate_transaction_data(
        num_records=50000,
        fraud_percentage=0.05,
        start_date='-60d'  # Extended to 60 days
    )
    
    # Analyze dataset
    analyze_dataset(transaction_df)
    
    # Save to CSV
    output_file = 'synthetic_mobile_money_transactions_enhanced.csv'
    transaction_df.to_csv(output_file, index=False)
    logger.info(f"Dataset saved to {output_file}")
    
    # Optional: Save a sample for quick inspection
    transaction_df.sample(1000).to_csv('sample_transactions.csv', index=False)