from transformers import pipeline

# 1. Load the pipeline (downloads model on first run)
generator = pipeline('text-generation', model='gpt2')

# 2. Query the model
outputs = generator("Hello, how are you today?", max_length=50, num_return_sequences=1)

# 3. Print result
print(outputs[0]['generated_text'])
