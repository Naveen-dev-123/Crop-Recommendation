from flask import Flask, flash,render_template,request
import numpy as np
import csv
import joblib
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
import  random

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result',methods=['POST','GET'])
def result():


    model = joblib.load('lightgbm_.pkl')

    classes = ['Paddy', 'Cholam', 'Cumbu', 'Ragi','Cotton', 'Sugarcane','Chilli', 'Pigeon Pea', 
                'Coconut', 'Tobacco', 'Onion', 'Banana','Mangoes', 'Turmeric', 'Groundnut', 'BlackGram', 
                'Maize', 'Tapioca','Tomoto', 'Brinjal', 'Carrot', 'Beans']

    values = []
    if request.method == 'POST':
        values.append(float(request.form.get('nitrogen')))
        values.append(float(request.form.get('phosphorous')))
        values.append(float(request.form.get('potassium')))
        values.append(float(request.form.get('temperature')))
        values.append(float(request.form.get('humidity')))
        values.append(float(request.form.get('ph')))
        values.append(float(request.form.get('rainfall')))

        answer = model.predict([values])
        predict_pro = model.predict_proba([values])
        list_proba = []
        for i in [-1, -2, -3, -4, -5]:
            list_proba.append(classes[np.argsort(np.max(predict_pro, axis=0))[i]])
        # print(list_proba)
        return render_template('result.html',probab = list_proba)

@app.route('/analysis')
def analysis():
    df = pd.read_csv('data.csv')
    def intractive_plot(df, feature, name):
    
        """
        This Function helps to create intractive Chart 
        ATTRIBUTE:
        df: original DataFrame
        feature: which column need to be 
        name : feature name
        """

        colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
               '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
               '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']

        df_label = pd.pivot_table(df, index=['label'], aggfunc='mean')
        df_label_feature = df_label.sort_values(by=feature, ascending = False)

        fig = make_subplots(rows = 1, cols = 2)

        top = {

            'y': df_label_feature[feature][:11].sort_values().index,
            'x': df_label_feature[feature][:11].sort_values()
        }
        last = {

            'y': df_label_feature[feature][-11:].sort_values().index,
            'x': df_label_feature[feature][-11:].sort_values()
        }

        fig.add_trace(
            go.Bar(top,
                   name='Highest {} Needed'.format(name),
                   marker_color = random.choice(colorarr),
                   orientation = 'h',
                   text = top['x']
                  ),
            row = 1, col = 1
        )
        fig.add_trace(
            go.Bar(last,
                   name='Least {} Needed'.format(name),
                   marker_color = random.choice(colorarr),
                   orientation = 'h',
                   text = top['x']
                  ),
            row = 1, col = 2
        )

        fig.update_traces(texttemplate = '%{text}', textposition = 'inside')
        fig.update_layout(title_text = name,
                          plot_bgcolor = 'white',
                          font_size = 12,
                          font_color = 'black',
                          height = 500
                         )


        fig.update_xaxes(showgrid = False)
        fig.update_yaxes(showgrid = False)
        fig.show()
    intractive_plot(df, feature = 'N', name = 'Nitrogen')
    intractive_plot(df, feature = 'P', name = 'Phosphorous')
    intractive_plot(df, feature = 'K', name = 'Potassium')
    intractive_plot(df, feature = 'humidity', name = 'Humidity')
    intractive_plot(df, feature = 'temperature', name = 'Temperature')
    intractive_plot(df, feature = 'ph', name = 'ph')
    intractive_plot(df, feature = 'rainfall', name = 'Rainfall')

@app.route('/real')
def real():
    return render_template('real.html')

@app.route('/l1')
def l1():
    l1 = [[68.6,2.2,50,33,60,8.4,67]]
    return render_template('predict1.html',data=l1,msg='Land 1')

@app.route('/l2')
def l2():
    l2 = [[75.6,3,58.3,34,61,7.8,70]]
    return render_template('predict1.html',data=l2,msg='Land 2')

@app.route('/l3')
def l3():
    l3 = [[71.4,4,70.8,33,62,7.9,81]]
    return render_template('predict1.html',data=l3,msg='Land 3')

@app.route('/l4')
def l4():
    l4 = [[62.2,4.2,66.6,35,59,7.7,110]]
    return render_template('predict1.html',data=l4,msg='Land 4')

@app.route('/Paddy-details')
def paddy_detail():
    return render_template('paddy_details.html')

@app.route('/Paddy-disease')
def paddy_disease():
    return render_template('paddy_disease.html')

@app.route('/Paddy-fertilizer')
def paddy_ferti():
    return render_template('paddy_ferti.html')

@app.route('/Cholam-details')
def cholam_detail():
    return render_template('cholam_details.html')

@app.route('/Cholam-disease')
def cholam_disease():
    return render_template('cholam_disease.html')

@app.route('/Cholam-fertilizer')
def cholam_ferti():
    return render_template('cholam_ferti.html')

@app.route('/Cumbu-details')
def cumbu_detail():
    return render_template('cumbu_details.html')

@app.route('/Cumbu-disease')
def cumbu_disease():
    return render_template('cumbu_disease.html')

@app.route('/Cumbu-fertilizer')
def cumbu_ferti():
    return render_template('cumbu_ferti.html')

@app.route('/Ragi-details')
def ragi_detail():
    return render_template('ragi_details.html')

@app.route('/Ragi-disease')
def ragi_disease():
    return render_template('ragi_disease.html')

@app.route('/Ragi-fertilizer')
def ragi_ferti():
    return render_template('ragi_ferti.html')

@app.route('/Cotton-details')
def cotton_detail():
    return render_template('cotton_details.html')

@app.route('/Cotton-disease')
def cotton_disease():
    return render_template('cotton_disease.html')

@app.route('/Cotton-fertilizer')
def cotton_ferti():
    return render_template('cotton_ferti.html')

@app.route('/Sugarcane-details')
def sugarcane_detail():
    return render_template('sugarcane_details.html')

@app.route('/Sugarcane-disease')
def sugarcane_disease():
    return render_template('sugarcane_disease.html')

@app.route('/Sugarcane-fertilizer')
def sugarcane_ferti():
    return render_template('sugarcane_ferti.html')

@app.route('/Chilli-details')
def chilli_detail():
    return render_template('chilli_details.html')

@app.route('/Chilli-disease')
def chilli_disease():
    return render_template('chilli_disease.html')

@app.route('/Chilli-fertilizer')
def chilli_ferti():
    return render_template('chilli_ferti.html')

@app.route('/Pigeon Pea-details')
def pigeon_detail():
    return render_template('pigeon_details.html')

@app.route('/Pigeon Pea-disease')
def pigeon_disease():
    return render_template('pigeon_disease.html')

@app.route('/Pigeon Pea-fertilizer')
def pigeon_ferti():
    return render_template('pigeon_ferti.html')

@app.route('/Coconut-details')
def coconut_detail():
    return render_template('coconut_details.html')

@app.route('/Coconut-disease')
def coconut_disease():
    return render_template('coconut_disease.html')

@app.route('/Coconut-fertilizer')
def coconut_ferti():
    return render_template('coconut_ferti.html')

@app.route('/Tobacco-details')
def tobacco_detail():
    return render_template('tobacco_details.html')

@app.route('/Tobacco-disease')
def tobacco_disease():
    return render_template('tobacco_disease.html')

@app.route('/Tobacco-fertilizer')
def tobacco_ferti():
    return render_template('tobacco_ferti.html')

@app.route('/Onion-details')
def onion_detail():
    return render_template('onion_details.html')

@app.route('/Onion-disease')
def onion_disease():
    return render_template('onion_disease.html')

@app.route('/Onion-fertilizer')
def onion_ferti():
    return render_template('onion_ferti.html')

@app.route('/Banana-details')
def banana_detail():
    return render_template('banana_details.html')

@app.route('/Banana-disease')
def banana_disease():
    return render_template('banana_disease.html')

@app.route('/Banana-fertilizer')
def banana_ferti():
    return render_template('banana_ferti.html')

@app.route('/Mangoes-details')
def mango_detail():
    return render_template('mango_details.html')

@app.route('/Mangoes-disease')
def mango_disease():
    return render_template('mango_disease.html')

@app.route('/Mangoes-fertilizer')
def mango_ferti():
    return render_template('mango_ferti.html')

@app.route('/Turmeric-details')
def termeric_detail():
    return render_template('termeric_details.html')

@app.route('/Turmeric-disease')
def termeric_disease():
    return render_template('termeric_disease.html')

@app.route('/Turmeric-fertilizer')
def termeric_ferti():
    return render_template('termeric_ferti.html')

@app.route('/Groundnut-details')
def ground_detail():
    return render_template('ground_details.html')

@app.route('/Groundnut-disease')
def ground_disease():
    return render_template('ground_disease.html')

@app.route('/Groundnut-fertilizer')
def ground_ferti():
    return render_template('ground_ferti.html')

@app.route('/BlackGram-details')
def black_detail():
    return render_template('black_details.html')

@app.route('/BlackGram-disease')
def black_disease():
    return render_template('black_disease.html')

@app.route('/BlackGram-fertilizer')
def black_ferti():
    return render_template('black_ferti.html')

@app.route('/Maize-details')
def maize_detail():
    return render_template('maize_details.html')

@app.route('/Maize-disease')
def maize_disease():
    return render_template('maize_disease.html')

@app.route('/Maize-fertilizer')
def maize_ferti():
    return render_template('maize_ferti.html')

@app.route('/Tapioca-details')
def topi_detail():
    return render_template('topi_details.html')

@app.route('/Tapioca-disease')
def topi_disease():
    return render_template('topi_disease.html')

@app.route('/Tapioca-fertilizer')
def topi_ferti():
    return render_template('topi_ferti.html')

@app.route('/Tomoto-details')
def tomoto_detail():
    return render_template('tomoto_details.html')

@app.route('/Tomoto-disease')
def tomoto_disease():
    return render_template('tomoto_disease.html')

@app.route('/Tomoto-fertilizer')
def tomoto_ferti():
    return render_template('tomoto_ferti.html')

@app.route('/Brinjal-details')
def brinjal_detail():
    return render_template('brin_details.html')

@app.route('/Brinjal-disease')
def brinjal_disease():
    return render_template('brin_disease.html')

@app.route('/Brinjal-fertilizer')
def brinjal_ferti():
    return render_template('brin_ferti.html')

@app.route('/Carrot-details')
def carrot_detail():
    return render_template('carrot_details.html')

@app.route('/Carrot-disease')
def carrot_disease():
    return render_template('carrot_disease.html')

@app.route('/Carrot-fertilizer')
def carrot_ferti():
    return render_template('carrot_ferti.html')

@app.route('/Beans-details')
def bean_detail():
    return render_template('bean_details.html')

@app.route('/Beans-disease')
def bean_disease():
    return render_template('bean_disease.html')

@app.route('/Beans-fertilizer')
def bean_ferti():
    return render_template('bean_ferti.html')

if __name__ == '__main__':
    app.run(debug=True)