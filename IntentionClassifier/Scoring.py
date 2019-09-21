import Results
from sklearn.metrics import confusion_matrix

def score(y_true, y_pred, labels):
    score = 0
    confusionMatrix = confusion_matrix(y_true, y_pred, labels=labels)
    respondLabels = ['Account', 'Authorisation', 'Availability', 'Colour', 
    'Admin', 'Documents', 'Project', 'EqualGlass', 'Gables', 'Leaver', 'Logo',
    'Pricing', 'Status', 'Weight', 'Access']

    #Predicted react, correctly reacted (TP): 1
    #Predicted react, incorrectly reacted (FP): -1
    #Predicted react, should have ignored (FP): -1
    #Predicted ignore, should have reacted (FN): 0
    #Predicted ignore, should have ignored (TN): 1
    for i in range(0, labels.count() - 1):
        for j in range(0, labels.count() - 1)
            if i = j:
                score += confusion_matrix[i][j]
            
