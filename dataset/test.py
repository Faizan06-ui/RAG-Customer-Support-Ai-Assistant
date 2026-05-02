from datasets import load_dataset

ds = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")

print("Features:", ds['train'].features)
print("First row:", ds['train'][0])