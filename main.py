import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import csv

# Load
train_data = pd.read_csv('E:/python_proj/DR_Proj/Final Project/training.csv')
test_data = pd.read_csv('E:/python_proj/DR_Proj/Final Project/testing.csv')


X_train, y_train = train_data.drop('Disease', axis=1), train_data['Disease']
X_test, y_test = test_data.drop('Disease', axis=1), test_data['Disease']

# Encode labels
le = LabelEncoder()
y_train, y_test = le.fit_transform(y_train), le.transform(y_test)

# Train
model = DecisionTreeClassifier().fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict disease
def predict_disease(symptoms_input):
    symptoms_input += [0] * (X_train.shape[1] - len(symptoms_input))  
    return le.inverse_transform(model.predict([symptoms_input]))[0]

# Load 
def predict_disease_from_csv(data, input_symptoms):
    input_symptoms = set(s.lower() for s in input_symptoms)
    best_match = None
    max_score = 0
    
    for row in data:
        disease = row[0]
        symptoms = set(s.lower() for s in row[1:7] if s) 
        score = len(input_symptoms & symptoms)  
        
        if score > max_score:
            max_score = score
            best_match = disease
    
    if best_match:
        return f"Predicted Disease: {best_match} (Matched {max_score} symptoms)"
    else:
        return "No matching disease found. Please check the symptoms entered."

# Main
def main():
    data = list(csv.reader(open(r'E:\python_proj\DR_Proj\Final Project\diseases_and_symptoms.csv')))
    
    print("Enter the Symptoms:- (Leave space for blank):")
    input_symptoms = []
    for i in range(5):  
        symptom = input(f"Symptom {i+1}: ").strip()
        if symptom:  
            
            
            input_symptoms.append(symptom)
        else:
            break  

    
    if input_symptoms:
        predicted_disease = predict_disease_from_csv(data, input_symptoms)
        print(predicted_disease)
    else:
        print("No symptoms entered. Cannot predict the disease.")
        

if __name__ == "__main__":
    main()



                