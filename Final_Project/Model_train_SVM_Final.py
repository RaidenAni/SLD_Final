import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 


data_dict = pickle.load(open('Final_Project\data.pickle_Final', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = SVC(kernel='linear')  

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')

print('Accuracy: {:.2f}%'.format(score * 100))
print('Precision: {:.2f}%'.format(precision * 100))
print('Recall: {:.2f}%'.format(recall * 100))
print('F1 score: {:.2f}%'.format(f1 * 100))

fig = go.Figure(data=[go.Table(
    header=dict(values=['Metric', 'Value (%)'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                       [f'{score * 100:.2f}', f'{precision * 100:.2f}', f'{recall * 100:.2f}', f'{f1 * 100:.2f}']],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(
    title_text='Model Evaluation Metrics',
    width=500, 
    height=300  
)

fig.show()  

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

f = open('./Final_Project/model_3.p', 'wb')  
pickle.dump({'model': model}, f)
f.close()