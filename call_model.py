from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Define the '/' root route to display the content from index.html
@app.route('/')
def home():
    return render_template('generator.html')


@app.route('/predict', methods=['POST'])
def predict():
    # form_data = allValues;
    # print(form_data)

    if request.method == 'POST':
        data = request.get_json()
        user_story = data.get('user_story')
        print("Received data from the front-end:", user_story)
        print("lala")

        model_path = "llama-2-7b-custom"
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Adjust the test User_Story to explicitly request the output structure
        User_Story = (
            "[INST]"+user_story+"[/INST]"
            "\n\nPlease provide the following:\n\n"
            "Scenario: Validate that the user can log in successfully.\n"
            "Given: The user launches and logs into the e-commerce application with <username> and <password>\n"
            "When: The user navigates to the account page.\n"
            "And: The user accesses the account dashboard.\n"
            "Then: The user should be able to view account details.\n"
        )

        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, device_map = 0)
        result = pipe(User_Story)

        # # Print the generated text, expecting the model to fill in the structured fields
        # print(result[0]['generated_text'])

        # Test with the generated text

        print(result[0]['generated_text'])



        render_template('generator.html')

        return jsonify({
            'user_story': str(result[0]['generated_text'])
        })

if __name__ == '__main__':
    app.run(debug=False)



