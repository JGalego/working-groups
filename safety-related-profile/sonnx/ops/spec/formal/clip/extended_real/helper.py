import math

# Define the possible values
options = [float('inf'), float('-inf'), float('nan'), 42, -17, 0]
labels = ['inf', '-inf', 'nan', 'real+', 'real-', 'zero']

def label(val):
    if math.isinf(val) and val > 0:
        return 'inf'
    elif math.isinf(val) and val < 0:
        return '-inf'
    elif math.isnan(val):
        return 'nan'
    elif val == 0:
        return 'zero'
    elif val > 0:
        return 'real+'
    elif val < 0:
        return 'real-'
    else:
        return 'real'

def operation(a, b):
    try:
        result = a / b
    except Exception as e:
        result = str(e)
    return result

print("Combination results for multiplication:")
for i, a in enumerate(options):
    for j, b in enumerate(options):
        res = operation(a, b)
        print(f"{labels[i]} / {labels[j]} = {label(res) if isinstance(res, float) else res}")