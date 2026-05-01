import time, urllib.request, urllib.parse, re

data = urllib.parse.urlencode({
    'borrower_name': 'John', 'loan_amnt': '100000', 'int_rate': '15',
    'installment': '3500', 'annual_inc': '50000', 'dti': '25',
    'fico_range_low': '580', 'fico_range_high': '600',
    'open_acc': '5', 'revol_bal': '8000', 'revol_util': '60',
    'total_acc': '12', 'delinq_2yrs': '1', 'inq_last_6mths': '2',
    'pub_rec': '0', 'bc_open_to_buy': '2000', 'bc_util': '70',
    'grade': 'D', 'purpose': 'debt_consolidation', 'term': '60 months',
}).encode()

start = time.time()
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data)
resp = urllib.request.urlopen(req)
body = resp.read().decode()
elapsed = time.time() - start

# Parse key values from HTML
probs = re.findall(r'([\d.]+)%', body)
risk = re.search(r'(LOW RISK|MEDIUM RISK|HIGH RISK|VERY HIGH RISK)', body)
amounts = re.findall(r'\$([\d,.\-]+)', body)

print(f"Response time: {elapsed*1000:.0f} ms")
print(f"Default probability: {probs[0]}%" if probs else "Prob: not found")
print(f"Risk level: {risk.group(0)}" if risk else "Risk: not found")
print(f"Financial values ($): {amounts}")
