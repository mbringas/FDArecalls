import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
import numpy as np
import statsmodels.api as sm
import pylab
import xgboost as xgb
!pip install matplotlib-venn
from matplotlib_venn import venn2, venn3

#Modelos y evaluadores
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score, f1_score, roc_auc_score, make_scorer


with open("/home/mauro/gitRepositories/foodAlertReports/food-enforcement-0001-of-0001.json",'r') as f:
    jsonread=json.load(f)

results=jsonread["results"]
meta = jsonread["meta"]
df = pd.DataFrame.from_records(results) 
print(df)
print(df.isna().sum())
df.initial_firm_notification.unique()

df.classification.unique()

plt.figure(figsize=(12,6))
nans=~df.isna().astype(int)
myColors = ((100, 0.0, 0.0, 1.0), (0, 100, 0, .60))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
ax = sns.heatmap(nans.T, cmap=cmap,cbar_kws={"shrink":0.3})
plt.ylabel("Nombre del campo")
plt.xlabel("Número de observación")
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([-1, -2])
colorbar.set_ticklabels(['Dato presente', 'Faltante'])
plt.show()




df.dtypes
df.columns
df.iloc[0,:]
df.product_quantity
df.product_type.value_counts()

#Fix dates
#Son strings de 8 caracteres YYYYMMDD
def put_together_dates_Series(dataSeries):
    year=dataSeries.str[0:4]
    month=dataSeries.str[4:6]
    day=dataSeries.str[6:]
    fin_date=pd.to_datetime(year+"-"+month+"-"+day)
    print(fin_date)
    return fin_date
df.columns

year_counts=df.recall_initiation_date.str[0:4].value_counts()
sns.barplot(year_counts,order=["0212"]+[str(i) for i in range(2008,2024)])
plt.xticks(rotation=90)
plt.xlabel("Fecha de iniciacion del reclamo")
plt.ylabel("Conteo")
plt.show()

newval=df.recall_initiation_date[df.recall_initiation_date.str[0:4]=="0212"].str.replace("0212","2012")
df.recall_initiation_date[df.recall_initiation_date.str[0:4]=="0212"]=newval

date_cols=["recall_initiation_date",'center_classification_date', 'report_date','termination_date']
for d in date_cols:
    df[d]=put_together_dates_Series(df[d])

df.isna().sum()

df.status.value_counts()

#Vemos conteo de las clases a clasificar
sns.countplot(data=df,x="classification",order=["Class I","Class II","Class III", "Not Yet Classified"])
for n,(lab,count) in enumerate(df.classification.value_counts().sort_index().items()):
    print(n,lab,count)
    if lab=="Not Yet Classified":
        lab="Not Yet\nClassified"
    plt.text(x=n-.2,y=count+1000,s=str(count))
plt.ylim((0,16000))
plt.ylabel("Cantidad")
plt.xlabel("Clasificación")
plt.show()


df["month_report"]=df.recall_initiation_date.dt.month
df["year_report"]=df.recall_initiation_date.dt.year
df["dow_report"]=df.recall_initiation_date.dt.dayofweek

#Clases según dia de la semana
por_diasem=df.value_counts(["classification","dow_report"]).sort_index().reset_index(level=[0,1])

plt.figure()
sns.barplot(data=por_diasem,x="dow_report",y="count",hue="classification")
#plt.legend(loc='center left' )#bbox_to_anchor=(1, 0.5))
plt.xticks(ticks=[0,1,2,3,4,5,6],labels=["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"],rotation=90)
plt.xlabel("Dia del inicio de reclamo")
plt.ylim(0,4000)
plt.ylabel("Cantidad")
plt.show()


#Clases según año
por_ano=df.value_counts(["classification","year_report"]).sort_index().reset_index(level=[0,1])

plt.figure()
sns.barplot(data=por_ano,x="year_report",y="count",hue="classification",legend=False)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Año del inicio de reporte")
plt.ylabel("Cantidad")
plt.xticks(rotation=90)
plt.show()

#Clases segun mes
por_mes=df.value_counts(["classification","month_report"]).sort_index().reset_index(level=[0,1])

plt.figure()
sns.barplot(data=por_mes,x="month_report",y="count",hue="classification",legend=False)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Mes del inicio de reclamo")
plt.ylabel("Cantidad")
plt.show()


#Tiempo hasta la clasificacion

df["classification_time"]=(df.center_classification_date-df.recall_initiation_date).dt.days
df["resolution_time"]=(df.termination_date-df.recall_initiation_date).dt.days


plt.figure(figsize=(9,3))
sns.histplot(data=df,x="classification_time")
plt.xlabel("Tiempo hasta la clasificacion (dias)")
plt.yscale("log")
plt.ylabel("Cantidad - escala logarítmica")
plt.show()


plt.figure(figsize=(9,3))
sns.histplot(data=df,x="classification_time",hue="classification",multiple="stack")
plt.xlabel("Tiempo hasta la clasificacion (dias)")
plt.yscale("log")
plt.ylabel("Cantidad - escala logarítmica")
plt.show()

df.classification_time.median()
(df.classification_time>90).sum()/df.classification_time.shape[0]
sns.histplot(data=df,x="classification_time",cumulative=True,fill=False)
plt.xlabel("Tiempo hasta la clasificacion (dias)")
plt.yscale("log")
plt.ylabel("Cantidad - escala logarítmica")
plt.show()
#Tiempo hasta la resolución


plt.figure(figsize=(9,3))
sns.histplot(data=df,x="resolution_time",hue="classification",multiple="stack")
plt.xlabel("Tiempo hasta la resolución (dias)")
plt.ylabel("Cantidad")
plt.show()

#Pruebas de normaldad
fig,ax=plt.subplots(nrows=3,ncols=1, figsize=(9,6))
for n in range(3):
    c=["Class I", "Class II", "Class III"][n]
    col=["blue","orange","red"][n]
    print(col)
    sns.histplot(data=df[df.classification==c],x="resolution_time",color=col,kde=True,ax=ax[n])
    median=df[df.classification==c].resolution_time.median()
    ax[n].axvline(x=median,ymax=10000,color=col,linestyle="--")
    ax[n].text(x=median+400,y=1100-n*500,s="Mediana de "+c+": "+str(median),c=col)
    ax[n].set_ylabel("Cantidad")
plt.xlabel("Tiempo de resolución")
plt.show()

#QQPLOTS
fig,ax=plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)
for n,c in enumerate(["Class I", "Class II", "Class III"]):
    sm.qqplot(df[df.classification==c].resolution_time, line='45',ax=ax[n])
    ax[n].set_title(c)
plt.show()

#KRUSKAL WALLIS para comparacion de medianas entre multiples grupos
from scipy.stats import kruskal

h_stat, p_value = kruskal(df[df.classification=="Class I"].resolution_time.dropna(), df[df.classification=="Class II"].resolution_time.dropna(), df[df.classification=="Class III"].resolution_time.dropna())
print(f"Kruskal-Wallis: H = {h_stat}, p-valor = {p_value}")

#Kruskal-Wallis: H = 66.20327804053947, p-valor = 4.2086308231356785e-15

from scikit_posthocs import posthoc_dunn

datos = [df[df.classification=="Class I"].resolution_time, df[df.classification=="Class II"].resolution_time,df[df.classification=="Class III"].resolution_time]

dunn_result = posthoc_dunn(datos, p_adjust='bonferroni')
print(dunn_result)


#counts_by_region

country_count=df.country.value_counts().reset_index()
state_count=df.groupby(["country"]).state.value_counts()
city_count=df.groupby(["country","state"]).city.value_counts()

#Bar chart of countries
sns.countplot(data=df,y="country",orient="h")
plt.xscale("log")
plt.ylim(2.5,-.5)
plt.ylabel("Pais")
plt.xlabel("Cantidad")
plt.show() 

#Column chart of states en EEUU
plt.figure(figsize=(10,4))
df_USA=df[df.country=="United States"]
sns.countplot(data=df_USA,x=df_USA.state,orient="v",order=df_USA.value_counts("state").sort_values(ascending=False).index)
plt.xticks(rotation=90)
plt.ylabel("Cantidad")
plt.xlabel("Estado de USA")
plt.show()
df_USA.state.value_counts()

#Mapa pintado con https://www.mapchart.net/usa.html

df.columns
#WordCloud de cities coloreados por Estado?
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
WordCloud().generate(' '.join(df['product_description']))
plt.show()

#count_by_year
df.report_date.dt.year.value_counts().plot(kind='bar')
plt.show()

#Identifico que solo hay un caso no clasificado
df["classification"].value_counts()
df=df[df.classification!="Not Yet Classified"].reset_index()

'''
#Extracción de cantidades a partir de la columna product quantity

df.product_quantity.unique()
quantity=df.product_quantity.str.split(" ",expand=True)

quantity.columns=[str(i) for i in range(38)]
quantity["0"].value_counts()
'''


lista=list(df.product_quantity.str.split())
text= " ".join([y for x in lista for y in x]).lower()
homogeneize={"cases":"case",
             "units":"unit",
             "lbs":"pound",
             "pounds":"pound",
             "lb":"pound",
             "ounces":"oz",
             "products":"product",
             "kilograms":"kg",
             "kgs":"kg",
             "kilogram":"kg"
    }
for o,r in homogeneize.items():
    text=text.replace(o,r)

from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=40,background_color="ivory",colormap="brg",collocations=False).generate(text)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Casos particulares
casos=["unit", "case", "pound","bottle", "package"]
for c in casos:
    s=df[df.product_quantity.str.contains(c)].product_quantity
    print(s.shape[0]/df.shape[0])
    numbers=s.str.extract("(\d+(?:[.,]\d+)*).*?"+c)
    numbers.columns=["amount"]
    numbers["amount"]=numbers.amount.str.replace(",","").astype(float)
    numbers["classification"]=df[df.product_quantity.str.contains(c)].classification
    print(max(numbers.amount))
    numbers.loc[numbers.amount>10000]=-1
    numbers=numbers[numbers.amount>-1]    
    plt.figure(figsize=(5,5))
    sns.histplot(data=numbers,x="amount",hue="classification",
                 hue_order = ['Class I','Class II','Class III'])
    plt.xlabel(f"Cantidad de producto para el término '{c}'")
    plt.ylabel("Cantidad de registros")
    histvals=np.histogram(numbers.amount,bins=1000)[0][0]
    plt.text(x=max(numbers.amount)//2,y=histvals//3,s="N = "+str(s.shape[0]))
    plt.show()

df.voluntary_mandated.value_counts()

#Al haber tanta variedad se me pierde la estructura
easy_quant=df.product_quantity.str.extract("(\d+\,?\.?\d+)")
easy_units=df.product_quantity.str.extract("\d+ (\w+)")
df["product_quant_number"]=easy_quant
df["product_quant_units"]=easy_units
df["product_quant_number"]=df.product_quant_number.str.replace(",","").fillna("-1").astype(float)
df["product_quant_units"]=df["product_quant_units"].fillna("")
df["product_quant_units"]=df["product_quant_units"].str.lower()


#Tomo las 20 unidades más frecuentes y las homogeneizo
list(df["product_quant_units"].value_counts().head(20).index)
df["product_quant_units"]=df["product_quant_units"].str.replace("gallons","gallon")
df["product_quant_units"]=df["product_quant_units"].str.replace("ounces","oz")
df["product_quant_units"]=df["product_quant_units"].str.replace("pounds","lbs")
df["product_quant_units"]=df["product_quant_units"].str.replace("lbs","lb")
df["product_quant_units"]=df["product_quant_units"].str.replace("kgs","kg")
for i in ["tubs","cartons","containers","boxes","packages","cs","bags","bottles","units","sandwiches","pieces","cans","jars",]:
    df["product_quant_units"]=df["product_quant_units"].str.replace(i,"cases")

retain_units=["cases","","lb","kg","tons","oz","gallon"]
def not_in(value):
    if value not in retain_units:
        return "unknown"
    else:
        return value
    
df["product_quant_units_clean"]=df["product_quant_units"].apply(not_in)
df["product_quant_units_clean"].value_counts()

units=list(df["product_quant_units_clean"].unique())
for u in units:
    df[df["product_quant_units_clean"]==u].product_quant_number.plot(kind="hist")
    plt.title(f"Unidad: {u}")
    plt.yscale('log')
    plt.ylim(0.1,100000)
    plt.show()

tons=df[df["product_quant_units_clean"]==u].product_quant_number

df.recalling_firm.value_counts().head(20)


#amount_of_recalls
df.recalling_firm.value_counts().plot(kind="hist")
plt.yscale("log")
plt.ylim(0,10000)
plt.show()
#
# Bivariado
#
df.columns

'''
---

### **b) Pregunta de clasificación con métodos de machine learning tradicionales**

Una pregunta interesante podría ser: **¿Podemos predecir la clasificación de un recall (`classification`) en función de las características del producto y el incidente?**

#### Pasos:
1. **Preprocesamiento:**
   - Codificar variables categóricas (`product_type`, `voluntary_mandated`, etc.) usando one-hot encoding o label encoding.
   - Escalar variables numéricas (`product_quantity`).
   - Extraer características de texto de `reason_for_recall` usando TF-IDF o embeddings.
   - Dividir el dataset en entrenamiento y prueba.

2. **Modelos tradicionales:**
   - Regresión Logística.
   - Support Vector Machines (SVM).
   - k-Nearest Neighbors (k-NN).

3. **Evaluación:**
   - Métricas como precisión, recall, F1-score y matriz de confusión.

'''

df["len_recall"]=df.reason_for_recall.str.len()
df["words_recall"]=df.reason_for_recall.str.lower().str.replace('\d+', '', regex = True).str.replace('[^\w\s\+]', '', regex = True).str.count(" ")+1

fig,ax=plt.subplots(1,2,figsize=(10,3))
sns.histplot(data=df,x="len_recall",hue="classification",ax=ax[0],bins=50)
ax[0].set_xlabel("Longitud del campo")
ax[0].set_ylabel("Cantidad")
sns.histplot(data=df,x="words_recall",hue="classification",ax=ax[1],bins=50)
ax[1].set_xlabel("Palabras")
ax[1].set_ylabel("Cantidad")
plt.show()

#Búsqueda de términos clave:

#Contaminantes inorgánicos
list_no_biologic={"lead","metal","plastic","pesticide","arsenic"}
df["no_biologic_matches"] = df["reason_for_recall"].str.lower().str.split().apply(set(list_no_biologic).intersection)
df["no_biologic_matches"] = df["no_biologic_matches"].apply(len)

no_biologic_counts=pd.crosstab(df.classification, df.no_biologic_matches)
sns.heatmap(no_biologic_counts,annot=True,fmt="d")
plt.xlabel("Cantidad de términos de interés")
plt.ylabel("Clasificación")
plt.show()
no_biologic_counts_percent=(no_biologic_counts.T*100/no_biologic_counts.sum(axis=1)).T
sns.heatmap(no_biologic_counts_percent,annot=True,fmt=".1f")
plt.xlabel("Cantidad de términos de interés")
plt.ylabel("Clasificación")
plt.show()

#Contaminantes biologicos
list_biologic={"bacteria","listeria","e. coli", "e coli","salmonella","monocytogenes","clostridium" ,"botulinum","hepatitis","mold","coliforms"}
df["biologic_matches"] = df["reason_for_recall"].str.lower().str.split().apply(set(list_biologic).intersection)
df["biologic_matches"] = df["biologic_matches"].apply(len)

biologic_counts=pd.crosstab(df.classification, df.biologic_matches)
sns.heatmap(biologic_counts,annot=True,fmt="d")
plt.xlabel("Cantidad de términos de interés")
plt.ylabel("Clasificación")
plt.show()
biologic_counts_percent=(biologic_counts.T*100/biologic_counts.sum(axis=1)).T
sns.heatmap(biologic_counts_percent,annot=True,fmt=".1f")
plt.xlabel("Cantidad de términos de interés")
plt.ylabel("Clasificación")
plt.show()

#Generamos datasets de train, val y test
#Vamos a usar train y val para elegir buenos hiperparámetros
#y haremos la evaluación final con test.


# Voy a probar un par de familias de modelos:
# - Naive bayes : útil para clasificacion de textos, conocida por su aplicacion
#   en deteccion de spam.

# - Regresor logístico: ajustar sus parámetros de regularizacion.
#   Le tengo desconfianza porque son muchas variables.

# - Random forest: Detectar 
    
# - XGboost

# - Red Neuronal



from sklearn.model_selection import train_test_split
#ver si se puede hacer estratificado


class_to_number={"Class I":1,"Class II":2,"Class III":3}
df["classification_number"]=df.classification.map(class_to_number)
Xint,Xtest,yint,ytest= train_test_split(df["reason_for_recall"],df["classification_number"],test_size=0.2,random_state=42)
Xtrain,Xval,ytrain,yval=train_test_split(Xint,yint,test_size=0.25,random_state=42)

train_idx=Xtrain.index.tolist()
val_idx=Xval.index.tolist()
test_idx=Xtest.index.tolist()

#Embeddings 1: Count vectorizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

def preprocess_cv(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def get_countVector(train,val,test):
    stop = stopwords.words('english')
    
    cv=CountVectorizer(analyzer='word',
                       stop_words=stop,
                       preprocessor=preprocess_cv,
                       binary=True,
                       lowercase=True,
                       ngram_range=(1,1)
                       )
    
    features_train=cv.fit_transform(train).toarray()
    features_test=cv.transform(test).toarray()
    features_val=cv.transform(val).toarray()
    
    return cv,features_train,features_val,features_test

countvectorizer,CVtrain,CVval,CVtest=get_countVector(Xtrain, Xval, Xtest)

def get_countVector(train,val,test):
    stop = stopwords.words('english')
    
    cv=CountVectorizer(analyzer='word',
                       stop_words=stop,
                       preprocessor=preprocess_cv,
                       binary=True,
                       lowercase=True,
                       ngram_range=(1,1)
                       )
    
    features_train=cv.fit_transform(train).toarray()
    features_test=cv.transform(test).toarray()
    features_val=cv.transform(val).toarray()
    
    return cv,features_train,features_val,features_test

tfidfvectorizer,TItrain,TIval,TItest=get_countVector(TItrain, TIval, Xtest)


#Embeddings 2: embeddings estaticos con word2vec
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
import nltk
import gensim.downloader as api
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Stopwords y lematizador
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#usamos el modelo GoogleNews 300
model = api.load("word2vec-google-news-300")
def get_sentence_embedding(sentence, model):
    # Preprocesamiento básico
    sentence = sentence.lower()
    sentence = re.sub(r'\d+', '', sentence)  # eliminar números
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # quitar puntuación
    
    words = word_tokenize(sentence)
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    word_embeddings = [model[word] for word in words if word in model]
    
    # Si no hay palabras válidas, devolver un vector de ceros
    if len(word_embeddings) == 0:
        response = np.zeros(model.vector_size)
    else:
        # Promediar los embeddings de las palabras
        response = np.mean(word_embeddings, axis=0)
    return response

def get_word2vec(train,val,test):
    sentence_embeddings = train.apply(lambda x: get_sentence_embedding(x, model))
    train_embeddings=np.stack(sentence_embeddings.values)
    sentence_embeddings = val.apply(lambda x: get_sentence_embedding(x, model))
    val_embeddings=np.stack(sentence_embeddings.values)
    sentence_embeddings = test.apply(lambda x: get_sentence_embedding(x, model))
    test_embeddings=np.stack(sentence_embeddings.values)
    return train_embeddings,val_embeddings,test_embeddings

W2Vtrain,W2Vval,W2Vtest=get_word2vec(Xtrain, Xval, Xtest)

#Embeddings 3: pre trained BERT

#esto se hizo una única vez en máquina de Google. Los embeddings los cargo ya calculados.
'''
#Usamos distilbert para multiclass clasification
# Usamos el BERT comun y entrenamos el clasificador posterior

!pip install transformers torch pandas scikit-learn
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Cargar el tokenizador y el modelo preentrenado
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Función para generar embeddings
def get_embeddings(texts, max_length=128):
    # Tokenizar el texto
    inputs = tokenizer(
        texts.tolist(),  # Convertir la columna de texto a una lista
        padding=True,    # Rellenar secuencias para que tengan la misma longitud
        truncation=True, # Truncar secuencias que excedan el máximo de tokens
        max_length=max_length,
        return_tensors='pt'  # Devolver tensores de PyTorch
    )
    
    # Obtener los embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Usar el embedding del token [CLS] (primera posición) como representación del texto
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings
'''

def load_bertEmbeddings(train_idx,val_idx,test_idx):
    embeddings=np.load("/home/mauro/gitRepositories/foodAlertReports/bertEmbeddings/embeddings_0.bin.npy")
    for i in range(1,27):
        tmp_embeddings=np.load(f"/home/mauro/gitRepositories/foodAlertReports/bertEmbeddings/embeddings_{i*1000}.bin.npy")
        embeddings=np.concatenate([embeddings,tmp_embeddings])
    print(embeddings.shape)
    return embeddings[train_idx],embeddings[val_idx],embeddings[test_idx]
    
Btrain,Bval,Btest=load_bertEmbeddings(train_idx, val_idx, test_idx)

best_f1score_models={}

### FUNCION PARA EL CV validation
def explorar_hiperparametros(clf, param_grid, X_train, y_train, X_val, y_val):
    roc_auc_weighted = make_scorer(roc_auc_score, multi_class='ovr', average='weighted', needs_proba=True)

    scoring = {
        'roc_auc_weighted': roc_auc_weighted,
        'f1_weighted': 'f1_weighted'
    }
    
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scoring,
                               refit="f1_weighted",verbose=1, n_jobs=-1)
    
    grid_search.fit(X_val, y_val)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=True)
    accuracy = accuracy_score(y_val, y_pred)
    
    print("Mejores parámetros:", grid_search.best_params_)
    print("Accuracy en el conjunto de prueba:", accuracy)
    print("\nReporte de clasificación:")
    print(classification_report(y_val, y_pred))
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "classification_report": report,
        "accuracy": accuracy,
        "best_model": best_model
    },grid_search


#MODELO 1 Naive bayes

f1_weighted_scorer = make_scorer(f1_score, average='weighted')
roc_auc_weighted = make_scorer(roc_auc_score, multi_class='ovr', average='weighted', needs_proba=True)

##Embeddings CV

p_cv = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
fit_predict=p_cv.fit(CVtrain,ytrain)
predictions=p_cv.predict(CVval)
probas=p_cv.predict_proba(CVval)

#en train
f1_weighted_scorer(p_cv,CVtrain,ytrain) #0.83
roc_auc_weighted(p_cv,CVtrain,ytrain) #0.93


#en test
f1_weighted_scorer(p_cv,CVtest,ytest) #0.82
roc_auc_weighted(p_cv,CVtest,ytest) #0.91


##Embeddings W2V
p_w2v = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
fit_predict=p_w2v.fit(W2Vtrain,ytrain)
predictions=p_w2v.predict(W2Vval)
probas=p_w2v.predict_proba(W2Vval)

#en train
f1_weighted_scorer(p_w2v,W2Vtrain,ytrain) #0.63
roc_auc_weighted(p_w2v,W2Vtrain,ytrain) #0.80

#en test
f1_weighted_scorer(p_w2v,W2Vtest,ytest) #0.65
roc_auc_weighted(p_w2v,W2Vtest,ytest) #0.80

##Embeddings DistilBert
p_db = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
fit_predict=p_db.fit(Btrain,ytrain)
predictions=p_db.predict(Bval)
probas=p_db.predict_proba(Bval)

#en train
f1_weighted_scorer(p_db,Btrain,ytrain) #0.68
roc_auc_weighted(p_db,Btrain,ytrain) #0.77

#en test
f1_weighted_scorer(p_db,Btest,ytest) #0.66
roc_auc_weighted(p_db,Btest,ytest) #0.78
    
#Con la 5-fold validation

mnb = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])

param_grid = {
    }

params_mnv_BoW,gridsearch_mnb_bow=explorar_hiperparametros(mnb, param_grid, CVtrain, ytrain, CVval, yval)
gridsearch_mnb_bow.cv_results_
params_mnv_W2V,gridsearch_mnb_w2v=explorar_hiperparametros(mnb, param_grid, W2Vtrain, ytrain, W2Vval, yval)
gridsearch_mnb_w2v.cv_results_
params_mnv_DB,gridsearch_mnb_db=explorar_hiperparametros(mnb, param_grid, Btrain, ytrain, Bval, yval)
gridsearch_mnb_db.cv_results_


#Modelo 2: Regresor logístico

lreg=LogisticRegression(solver="saga")
param_grid = {
    'penalty': ["l1","l2"],
    'C': [1e-4,1e-3,1e-2,1e-1,1]
    }

#Modelo Bag of Words
params_logisticRegression_BoW,gridsearch_lr_bow=explorar_hiperparametros(lreg, param_grid, CVtrain, ytrain, CVval, yval)

params_logisticRegression_W2V,gridsearch_lr_w2v=explorar_hiperparametros(lreg, param_grid, W2Vtrain, ytrain, W2Vval, yval)

params_logisticRegression_dB,gridsearch_lr_db=explorar_hiperparametros(lreg, param_grid, Btrain, ytrain, Bval, yval)

######################
#Armo las curvas de accuracy en función del parámetro C con todos los embeddings
results=[]
for x,y,t in zip([CVtrain,W2Vtrain,Btrain],[ytrain,ytrain,ytrain],[CVtest,W2Vtest,Btest]):
    results.append([])
    for pen in ["l1","l2"]:
        results[-1].append({pen:{}})
        for C in [1e-4,1e-3,1e-2,1e-1,1]:
            lreg=LogisticRegression(solver="saga",penalty=pen,C=C)
            lreg.fit(x,y)
            predicts=lreg.predict(t)
            print("listo")
            results[-1][-1][pen][C]=accuracy_score(ytest,predicts)
results
print(results)
results_df = pd.DataFrame(results)
colors=["b","r","k"]
labels=["Bag of Words","Word2Vec","distil Bert"]
for n,datx in enumerate(results):
    xl1=datx[0]["l1"].keys()
    print(np.array(list(datx[0]["l1"].values())))
    yl1=np.array(list(datx[0]["l1"].values()))*100
    yl2=np.array(list(datx[1]["l2"].values()))*100
    plt.plot(xl1,yl1,"-",c=colors[n],label=f"{labels[n]} regularización L1")
    plt.plot(xl1,yl2,"--",c=colors[n],label=f"{labels[n]} regularización L2")
plt.title("Exploración de hiperparámetros\nRegresión Logística")
plt.xscale("log")
plt.xlabel("Coeficiente de regularización")
plt.legend(loc="lower right")
plt.ylim(0,100)
plt.ylabel("Accuracy en test")
plt.show()

######################

#Modelo 3: Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=42,class_weight="balanced")
param_grid = {
    "n_estimators":[10,50,100,250,500,1000],
    'max_depth': [3,5,10]
    }

params_randForest_BoW,gridsearch_rf_bow=explorar_hiperparametros(rfc, param_grid, CVtrain, ytrain, CVval, yval)

params_randForest_W2V,gridsearch_rf_w2v=explorar_hiperparametros(rfc, param_grid, W2Vtrain, ytrain, W2Vval, yval)

params_randForest_dB,gridsearch_rf_db=explorar_hiperparametros(rfc, param_grid, Btrain, ytrain, Bval, yval)

######################
#Armo las curvas de accuracy en función de n_estimators y max_depth
results_rf=[]
for x,y,t in zip([CVtrain,W2Vtrain,Btrain],[ytrain,ytrain,ytrain],[CVtest,W2Vtest,Btest]):
    results_rf.append({})
    for n_est in [10,25,50,75,100,150,200]:
        results_rf[-1][n_est]={}
        for maxdep in [3,5,8,10,15,20]:
            rfc=RandomForestClassifier(random_state=42,class_weight="balanced",max_depth=maxdep,n_estimators=n_est)
            rfc.fit(x,y)
            predicts=rfc.predict(t)
            print("listo")
            results_rf[-1][n_est][maxdep]=accuracy_score(ytest,predicts)
results_rf
results_rf_df={"Bag of Words":results_rf[0],"Word2Vec":results_rf[1],"DistilBert":results_rf[1]}

rows = []
for modelo, samples in results_rf_df.items():
    for sample_size, params in samples.items():
        for param, valor in params.items():
            rows.append({
                "embedding": modelo,
                "n_estimators": sample_size,
                "max_depth": param,
                "accuracy": valor*100
            })

results_rf_df = pd.DataFrame(rows)

colors=["Blues","Reds","Greys"]
fig,ax=plt.subplots(3,1,figsize=(6,8),sharex=True,sharey=True)
for n,model in enumerate(results_rf_df.modelo.unique()):
    sns.barplot(data=results_rf_df,x="n_estimators",y="accuracy",hue="max_depth",ax=ax[n],palette=colors[n],ci=None,edgecolor="black", linewidth=.5)
    ax[n].set_title(f"{embedding}")
ax[0].legend_.remove()
ax[1].legend_.remove()
ax[2].legend_.remove()
ax[0].set_ylim(50,90)
ax[1].set_ylim(50,90)
ax[2].set_ylim(50,90)
handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
           
fig.xlabel("Máxima profundidad")
fig.legend(loc="lower right")
fig.ylabel("Accuracy en test")
plt.show()

sns.lineplot(data=results_rf_df,x="n_estimators",y="accuracy",hue="max_depth",style="embedding",palette="turbo",markers=True)
plt.xticks([10,25,50,75,100,150,200])
plt.ylim(45,95)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

######################
'''

#Modelo 4: Single layer perceptron
from sklearn.linear_model import Perceptron

##Embeddings CV
pct_cv = Pipeline([('Normalizing',MinMaxScaler()),('PerceptronCapaSimple',Perceptron())])
fit_predict=pct_cv.fit(CVtrain,ytrain)
predictions=pct_cv.predict(CVval)
#accuracy global
sum(pct_cv.predict(CVtrain)==ytrain)/len(ytrain)
#0.91
sum(predictions==yval)/len(predictions)
#0.85

##Embeddings W2V
pct_w2v = Pipeline([('Normalizing',MinMaxScaler()),('PerceptronCapaSimple',Perceptron())])
fit_predict=pct_w2v.fit(W2Vtrain,ytrain)
predictions=pct_w2v.predict(W2Vval)
#accuracy global
sum(pct_w2v.predict(W2Vtrain)==ytrain)/len(ytrain)
#0.745
sum(predictions==yval)/len(predictions)
#0.743

##Embeddings DistilBert
pct_db = Pipeline([('Normalizing',MinMaxScaler()),('PerceptronCapaSimple',Perceptron())])
fit_predict=pct_db.fit(Btrain,ytrain)
predictions=pct_db.predict(Bval)
#accuracy global
sum(pct_db.predict(Btrain)==ytrain)/len(ytrain)
#0.785
sum(predictions==yval)/len(predictions)
#0.702
'''
#Modelo 5: XGBoost


param_grid = {
    "n_estimators":[100,200,300],
    'learning_rate': [.01,.1],
    "max_depth":[3,6,9],
    "subsample":[0.7,0.8],
    "colsample_bytree":[0.7,.8]
    }

param_grid = {
    "n_estimators":[100,200,300],
    'learning_rate': [.01,.1],
    "max_depth":[3,6,9],
    "subsample":[0.7,0.8],
    "colsample_bytree":[0.7,.8]
    }

xgbmod=xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=3, 
                            random_state=42)

##Embeddings CV
params_xgb_BoW,gridsearch_xgb_bow=explorar_hiperparametros(xgbmod, 
                                                           param_grid, 
                                                           CVtrain, 
                                                           ytrain-1, 
                                                           CVval, 
                                                           yval-1)

'''{'colsample_bytree': 0.7,
 'learning_rate': 0.1,
 'max_depth': 6,
 'n_estimators': 300,
 'subsample': 0.7}
    
xgbmod=xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=3, 
                            random_state=42,colsample_bytree= 0.7,
                             learning_rate= 0.1,
                             max_depth= 6,
                             n_estimators= 300,
                             subsample =0.7)    
xgbmod.fit(CVtrain,ytrain-1)
predicts=xgbmod.predict(CVtest)
accuracy_cv = accuracy_score(predicts,ytest-1)

    '''


##Embeddings W2V

params_xgb_W2V,gridsearch_xgb_W2V=explorar_hiperparametros(xgbmod, 
                                                           param_grid, 
                                                           W2Vtrain, 
                                                           ytrain-1, 
                                                           W2Vval, 
                                                           yval-1)

'''{'colsample_bytree': 0.8,
 'learning_rate': 0.1,
 'max_depth': 6,
 'n_estimators': 300,
 'subsample': 0.7}
    
    
xgbmod=xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=3, 
                            random_state=42,colsample_bytree= 0.8,
                             learning_rate= 0.1,
                             max_depth= 6,
                             n_estimators= 300,
                             subsample =0.7)    
xgbmod.fit(W2Vtrain,ytrain-1)
predicts=xgbmod.predict(W2Vtest)
accuracy_w2v = accuracy_score(predicts,ytest-1)
    
    '''

##Embeddings DistilBert

params_xgb_DB,gridsearch_xgb_DB=explorar_hiperparametros(xgbmod, 
                                                           param_grid, 
                                                           Btrain, 
                                                           ytrain-1, 
                                                           Bval, 
                                                           yval-1)
'''
{'colsample_bytree': 0.7,
 'learning_rate': 0.1,
 'max_depth': 6,
 'n_estimators': 300,
 'subsample': 0.7}


xgbmod=xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=3, 
                            random_state=42,colsample_bytree= 0.7,
                             learning_rate= 0.1,
                             max_depth= 6,
                             n_estimators= 300,
                             subsample =0.7)    
xgbmod.fit(Btrain,ytrain-1)
predicts=xgbmod.predict(Btest)
accuracy_db = accuracy_score(predicts,ytest-1)

'''



######################
#Armo las curvas de accuracy en función de n_estimators y max_depth
results_rf=[]
for x,y,t in zip([CVtrain,W2Vtrain,Btrain],[ytrain,ytrain,ytrain],[CVtest,W2Vtest,Btest]):
    results_rf.append({})
    for n_est in [100,200,300]:
        results_rf[-1][n_est]={}
        for maxdep in [3,5,8,10,15,20]:
            rfc=RandomForestClassifier(random_state=42,class_weight="balanced",max_depth=maxdep,n_estimators=n_est)
            rfc.fit(x,y)
            predicts=rfc.predict(t)
            print("listo")
            results_rf[-1][n_est][maxdep]=accuracy_score(ytest,predicts)
results_rf
results_rf_df={"Bag of Words":results_rf[0],"Word2Vec":results_rf[1],"DistilBert":results_rf[1]}

rows = []
for modelo, samples in results_rf_df.items():
    for sample_size, params in samples.items():
        for param, valor in params.items():
            rows.append({
                "embedding": modelo,
                "n_estimators": sample_size,
                "max_depth": param,
                "accuracy": valor*100
            })

results_rf_df = pd.DataFrame(rows)

colors=["Blues","Reds","Greys"]
fig,ax=plt.subplots(3,1,figsize=(6,8),sharex=True,sharey=True)
for n,model in enumerate(results_rf_df.modelo.unique()):
    sns.barplot(data=results_rf_df,x="n_estimators",y="accuracy",hue="max_depth",ax=ax[n],palette=colors[n],ci=None,edgecolor="black", linewidth=.5)
    ax[n].set_title(f"{embedding}")
ax[0].legend_.remove()
ax[1].legend_.remove()
ax[2].legend_.remove()
ax[0].set_ylim(50,90)
ax[1].set_ylim(50,90)
ax[2].set_ylim(50,90)
handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
           
fig.xlabel("Máxima profundidad")
fig.legend(loc="lower right")
fig.ylabel("Accuracy en test")
plt.show()

sns.lineplot(data=results_rf_df,x="n_estimators",y="accuracy",hue="max_depth",style="embedding",palette="turbo",markers=True)
plt.xticks([10,25,50,75,100,150,200])
plt.ylim(45,95)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

######################



#Evaluacion en Test
#Armamos unas matrices de accuracy en test

acc={}
f1_weighted_scorer(gridsearch_mnb_bow.best_estimator_,CVtest,ytest)
f1_weighted_scorer(gridsearch_mnb_w2v.best_estimator_,W2Vtest,ytest)
f1_weighted_scorer(gridsearch_mnb_db.best_estimator_,Btest,ytest)

roc_auc_weighted(gridsearch_mnb_bow.best_estimator_,CVtest,ytest)
roc_auc_weighted(gridsearch_mnb_w2v.best_estimator_,W2Vtest,ytest)
roc_auc_weighted(gridsearch_mnb_db.best_estimator_,Btest,ytest)

acc[("nb","cv")]=sum(p_cv.predict(CVtest)==ytest)/len(ytest)
acc[("nb","w2v")]=sum(p_w2v.predict(W2Vtest)==ytest)/len(ytest)
acc[("nb","db")]=sum(p_db.predict(Btest)==ytest)/len(ytest)

#Logistic Regression

f1_weighted_scorer(gridsearch_lr_bow.best_estimator_,CVtest,ytest)
f1_weighted_scorer(gridsearch_lr_w2v.best_estimator_,W2Vtest,ytest)
f1_weighted_scorer(gridsearch_lr_db.best_estimator_,Btest,ytest)

roc_auc_weighted(gridsearch_lr_bow.best_estimator_,CVtest,ytest)
roc_auc_weighted(gridsearch_lr_w2v.best_estimator_,W2Vtest,ytest)
roc_auc_weighted(gridsearch_lr_db.best_estimator_,Btest,ytest)

acc[("lr","cv")]=sum(gridsearch_lr_bow.best_estimator_.predict(CVtest)==ytest)/len(ytest)
acc[("lr","w2v")]=sum(gridsearch_lr_w2v.best_estimator_.predict(W2Vtest)==ytest)/len(ytest)
acc[("lr","db")]=sum(gridsearch_lr_db.best_estimator_.predict(Btest)==ytest)/len(ytest)


f1_weighted_scorer(gridsearch_rf_bow.best_estimator_,CVtest,ytest)
f1_weighted_scorer(gridsearch_rf_w2v.best_estimator_,W2Vtest,ytest)
f1_weighted_scorer(gridsearch_rf_db.best_estimator_,Btest,ytest)

roc_auc_weighted(gridsearch_rf_bow.best_estimator_,CVtest,ytest)
roc_auc_weighted(gridsearch_rf_w2v.best_estimator_,W2Vtest,ytest)
roc_auc_weighted(gridsearch_rf_db.best_estimator_,Btest,ytest)

acc[("rf","cv")]=sum(gridsearch_rf_bow.best_estimator_.predict(CVtest)==ytest)/len(ytest)
acc[("rf","w2v")]=sum(gridsearch_rf_w2v.best_estimator_.predict(W2Vtest)==ytest)/len(ytest)
acc[("rf","db")]=sum(gridsearch_rf_db.best_estimator_.predict(Btest)==ytest)/len(ytest)


f1_weighted_scorer(gridsearch_xgb_bow.best_estimator_,CVtest,ytest)
f1_weighted_scorer(gridsearch_xgb_W2V.best_estimator_,W2Vtest,ytest)
f1_weighted_scorer(gridsearch_xgb_DB.best_estimator_,Btest,ytest)

roc_auc_weighted(gridsearch_xgb_bow.best_estimator_,CVtest,ytest)
roc_auc_weighted(gridsearch_xgb_W2V.best_estimator_,W2Vtest,ytest)
roc_auc_weighted(gridsearch_xgb_DB.best_estimator_,Btest,ytest)

acc[("slp","cv")]=sum(pct_cv.predict(CVtest)==ytest)/len(ytest)
acc[("slp","w2v")]=sum(pct_w2v.predict(W2Vtest)==ytest)/len(ytest)
acc[("slp","db")]=sum(pct_db.predict(Btest)==ytest)/len(ytest)

#buscamos la proba correcta para los distintos thresholds

import numpy as np
from sklearn.metrics import f1_score
from itertools import product
import matplotlib.pyplot as plt

def find_best_thresholds(y_true, y_probs, step=5):
    thresholds = np.arange(0, 100, step)/100
    best_score = -1
    best_thresh = None
    scores = []

    for thresh_comb in product(thresholds, repeat=3):
        y_pred = custom_argmax(y_probs, thresh_comb)
        score = f1_score(y_true, y_pred, average='weighted')
        scores.append((thresh_comb, score))
        if score > best_score:
            best_score = score
            best_thresh = thresh_comb

    return best_thresh, best_score, scores

def custom_argmax(probs, thresholds):
    adjusted = probs - thresholds  # penalizamos por debajo del threshold
    return np.argmax(adjusted, axis=1)

import seaborn as sns
import pandas as pd

def plot_threshold_heatmap(scores,clasificador,embedding):
    
    data = pd.DataFrame([
        {'Threshold_clase1': t[0], 'Threshold_clase2': t[1], 'Score': s}
        for (t, s) in scores
    ])

    pivot = data.pivot(index='Threshold_clase1', columns='Threshold_clase2', values='Score')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f'F1 weighted - Threshold  de clases \n Clasificador {clasificador} Embedding {embedding}')
    plt.xlabel('Threshold Clase 1')
    plt.ylabel('Threshold Clase 2')
    plt.show()


def get_thresholds_and_calculate_matrix(gridsearch_object,train,ytrain,test,ytest,color,model,embedding):
    probas=gridsearch_object.best_estimator_.predict_proba(train)
    best_thresh, best_score, all_scores = find_best_thresholds(ytrain-1, probas)
    
    print("Mejores thresholds:", best_thresh)
    print("F1 score óptimo:", best_score)
    
    #Evaluacion del test
    probas=gridsearch_object.best_estimator_.predict_proba(test)
    y_pred_thresh = custom_argmax(probas, best_thresh)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(ytest, y_pred_thresh+1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=color, values_format="d")
    plt.title(f"Matriz de Confusión con Umbrales Óptimos\n {model} \n Embedding {embedding}")
    plt.xticks(ticks=[0,1,2],labels=["Clase I","Clase II","Clase III"])
    plt.yticks(ticks=[0,1,2],labels=["Clase I","Clase II","Clase III"],rotation=90)
    plt.xlabel("Predicho")
    plt.ylabel("Etiqueta real")
    plt.show()
    
get_thresholds_and_calculate_matrix(gridsearch_mnb_bow,CVtrain,ytrain,CVtest,ytest,"Blues","NB Multinomial","Bag of Words")
get_thresholds_and_calculate_matrix(gridsearch_mnb_w2v,W2Vtrain,ytrain,W2Vtest,ytest,"Greens","NB Multinomial","Word2Vec")
get_thresholds_and_calculate_matrix(gridsearch_mnb_db,Btrain,ytrain,Btest,ytest,"Reds","NB Multinomial","distilBert")

get_thresholds_and_calculate_matrix(gridsearch_lr_bow,CVtrain,ytrain,CVtest,ytest,"Blues","Regresión Logística","Bag of Words")
get_thresholds_and_calculate_matrix(gridsearch_lr_w2v,W2Vtrain,ytrain,W2Vtest,ytest,"Greens","Regresión Logística","Word2Vec")
get_thresholds_and_calculate_matrix(gridsearch_lr_db,Btrain,ytrain,Btest,ytest,"Reds","Regresión Logística","distilBert")

get_thresholds_and_calculate_matrix(gridsearch_rf_bow,CVtrain,ytrain,CVtest,ytest,"Blues","Random Forest","Bag of Words")
get_thresholds_and_calculate_matrix(gridsearch_rf_w2v,W2Vtrain,ytrain,W2Vtest,ytest,"Greens","Random Forest","Word2Vec")
get_thresholds_and_calculate_matrix(gridsearch_rf_db,Btrain,ytrain,Btest,ytest,"Reds","Random Forest","distilBert")

get_thresholds_and_calculate_matrix(gridsearch_xgb_bow,CVtrain,ytrain,CVtest,ytest,"Blues","XGBoost","Bag of Words")
get_thresholds_and_calculate_matrix(gridsearch_xgb_W2V,W2Vtrain,ytrain,W2Vtest,ytest,"Greens","XGBoost","Word2Vec")
get_thresholds_and_calculate_matrix(gridsearch_xgb_DB,Btrain,ytrain,Btest,ytest,"Reds","XGBoost","distilBert")


mpg=np.zeros((3,4))
mpg[(0,0)]=acc[("nb","cv")]
mpg[(1,0)]=acc[("nb","w2v")]
mpg[(2,0)]=acc[("nb","cv")]
mpg[(0,1)]=acc[("lr","cv")]
mpg[(1,1)]=acc[("lr","w2v")]
mpg[(2,1)]=acc[("lr","db")]



mpg[(0,2)]=acc[("rf","cv")]
mpg[(1,2)]=acc[("rf","w2v")]
mpg[(2,2)]=acc[("rf","db")]
mpg[(0,3)]=acc[("slp","cv")]
mpg[(1,3)]=acc[("slp","w2v")]
mpg[(2,3)]=acc[("slp","db")]

xlabs=["Naive Bayes","Logistic Regression","Random Forest","Single Layer Perceptron"]
ylabs=["CountVectorizer","Word2Vec","distilBert"]
sns.heatmap(mpg,vmin=0,vmax=1,annot=True,cmap = 'RdYlGn',xticklabels=xlabs,yticklabels=ylabs)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

#Matrices de confusión

xtick=["Clase I","Clase II","Clase III"]
ytick=["Clase I","Clase II","Clase III"]

def plot_matrices(matriz):
    fig, ax = plt.subplots(1, 3,sharex=True,sharey=True,figsize=(13,3))
    plt.text(-10,2.5,s="Clase predicha",rotation=90,fontsize=17)
    plt.text(-3,4,s="Clase real",fontsize=17)
    sns.heatmap(matriz[0], vmin=0,vmax=len(ytest),cmap="Blues",annot=True,xticklabels=xtick,yticklabels=ytick,ax=ax[0])        
    ax[0].set_title("Bag of Words")
    sns.heatmap(matriz[1], vmin=0,vmax=len(ytest),cmap="Reds",annot=True,xticklabels=xtick,yticklabels=ytick,ax=ax[1])
    ax[1].set_title("Word2Vec")
    sns.heatmap(matriz[2], vmin=0,vmax=len(ytest),cmap="Greens",annot=True,xticklabels=xtick,yticklabels=ytick,ax=ax[2])        
    ax[2].set_title("Distil BERT")
    plt.show()
    return None

#NaiveBayes
matrices=[]

for preds in [p_cv.predict(CVtest),p_w2v.predict(W2Vtest),p_db.predict(Btest)]:
    matrices.append(np.zeros((3,3)))
    for p,a in zip(preds,ytest):
        print(p,a)
        matrices[-1][(p-1,a-1)]+=1
        
plot_matrices(matrices)

#LogisticRegression
matrices=[]

for preds in [gridsearch_lr_bow.best_estimator_.predict(CVtest),gridsearch_lr_w2v.best_estimator_.predict(W2Vtest),gridsearch_lr_db.best_estimator_.predict(Btest)]:
    matrices.append(np.zeros((3,3)))
    for p,a in zip(preds,ytest):
        print(p,a)
        matrices[-1][(p-1,a-1)]+=1
        
plot_matrices(matrices)


#Random Forest
matrices=[]

for preds in [gridsearch_rf_bow.best_estimator_.predict(CVtest),gridsearch_rf_w2v.best_estimator_.predict(W2Vtest),gridsearch_rf_db.best_estimator_.predict(Btest)]:
    matrices.append(np.zeros((3,3)))
    for p,a in zip(preds,ytest):
        print(p,a)
        matrices[-1][(p-1,a-1)]+=1
        
plot_matrices(matrices)


#Single Layer Perceptron
matrices=[]

for preds in [pct_cv.predict(CVtest),pct_w2v.predict(W2Vtest),pct_db.predict(Btest)]:
    matrices.append(np.zeros((3,3)))
    for p,a in zip(preds,ytest):
        print(p,a)
        matrices[-1][(p-1,a-1)]+=1
        
plot_matrices(matrices)




'''
Acá definimos algunas reducciones de dimensionalidad para embeddings para ver si 
se agrupan de alguna forma los registros de distintas clases.

'''


#Dimensionality reduction de los embeddings
#A) PCA
from sklearn.decomposition import PCA, SparsePCA
def get_pca(features,labels,ncomps=3,sparse=0):
    pca=PCA(n_components=ncomps)
    if sparse==1:
        pca =SparsePCA(n_components=ncomps)
    proyecciones=pca.fit_transform(features)
    return pca,proyecciones

#BoW
n_pcas=500
pca_cvembedding,cv_projections=get_pca(CVtrain,ytrain,n_pcas)
varianza_expl=pca_cvembedding.explained_variance_ratio_
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.bar(range(n_pcas),varianza_expl*100)
ax1.set_xlabel("Componente Principal",fontsize=14)
ax1.set_ylabel("Varianza de la componente principal",fontsize=14)
ax1.set_ylim(0,max(varianza_expl*100)*1.05)
#plt.show()
cum_variance=np.cumsum(varianza_expl)*100
ax2.plot(range(n_pcas), cum_variance,color="r")
ax2.set_ylim(0,100)
ax2.set_ylabel("Varianza acumulada", fontsize=14)
ax2.axhline(y=max(cum_variance),color="r",linestyle="--")
ax2.axhline(y=50,color="r",linestyle="--")
ax2.text(n_pcas//2,max(cum_variance)*0.9,f"Acumula {round(max(cum_variance),2)}%",fontsize=18)
plt.show()

#W2V
pcnumb=300
pca_w2vembedding,w2v_projections=get_pca(W2Vtrain,ytrain,pcnumb)
varianza_expl=pca_w2vembedding.explained_variance_ratio_
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.bar(range(pcnumb),varianza_expl*100)
ax1.set_xlabel("Componente Principal",fontsize=14)
ax1.set_ylabel("Varianza de la componente principal",fontsize=14)
ax1.set_ylim(0,max(varianza_expl*100)*1.05)
#plt.show()
cum_variance=np.cumsum(varianza_expl)*100
ax2.plot(range(pcnumb), cum_variance,color="r")
ax2.set_ylim(0,100)
ax2.set_ylabel("Varianza acumulada", fontsize=14)
ax2.axhline(y=max(cum_variance),color="r",linestyle="--")
ax2.axhline(y=50,color="r",linestyle="--")
ax2.text(pcnumb//2,max(cum_variance)*0.9,f"Acumula {round(max(cum_variance),2)}%",fontsize=18)
plt.show()

#dB
pcnumb=768
pca_dbembedding,db_projections=get_pca(Btrain,ytrain,pcnumb)
varianza_expl=pca_dbembedding.explained_variance_ratio_
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.bar(range(pcnumb),varianza_expl*100)
ax1.set_xlim(0,pcnumb)
ax1.set_xlabel("Componente Principal",fontsize=14)
ax1.set_ylabel("Varianza de la componente principal",fontsize=14)
ax1.set_ylim(0,max(varianza_expl*100)*1.05)
#plt.show()
cum_variance=np.cumsum(varianza_expl)*100
ax2.plot(range(pcnumb), cum_variance,color="r")
ax2.set_ylim(0,100)
ax2.set_xlim(0,pcnumb)
ax2.set_ylabel("Varianza acumulada", fontsize=14)
ax2.axhline(y=max(cum_variance),color="r",linestyle="--")
ax2.axhline(y=50,color="r",linestyle="--")
ax2.text(pcnumb//2,max(cum_variance)*0.9,f"Acumula {round(max(cum_variance),2)}%",fontsize=18)
plt.show()

#Graficamos las primeras 2 PC
fig, ax = plt.subplots(2, 3,figsize=(8,6),sharex=True,sharey=True)
ax[0,0].scatter(cv_projections[:,0],cv_projections[:,1],c=ytrain,alpha=0.5,s=0.1)
ax[0,0].set_ylabel("PCA1", fontsize=16)
ax[0,0].set_title("Bag of Words", fontsize=16)
ax[1,0].scatter(cv_projections[:,0],cv_projections[:,2],c=ytrain,alpha=0.5,s=0.1)
ax[1,0].set_xlabel("PCA0", fontsize=16)
ax[1,0].set_ylabel("PCA2", fontsize=16)

ax[0,1].set_title("Word2Vec", fontsize=16)
ax[0,1].scatter(w2v_projections[:,0],w2v_projections[:,1],c=ytrain,alpha=.5,s=0.1)
ax[1,1].scatter(w2v_projections[:,0],w2v_projections[:,2],c=ytrain,alpha=0.5,s=0.1)
ax[1,1].set_xlabel("PCA0", fontsize=16)

ax[0,2].set_title("Distil BERT", fontsize=16)
ax[0,2].scatter(db_projections[:,0],db_projections[:,1],c=ytrain,alpha=0.5,s=0.1)
ax[1,2].scatter(db_projections[:,0],db_projections[:,2],c=ytrain,alpha=0.5,s=0.1)
ax[1,2].set_xlabel("PCA0", fontsize=16)
ax[1,2].legend()
plt.show()


fig, ax = plt.subplots()
scatter = ax.scatter(cv_projections[:,0],cv_projections[:,1],c=ytrain,alpha=0.5,s=0.1)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
plt.show()

#Sparse PCA

#BoW
pca_cvembedding,cv_projections=get_pca(CVtrain,ytrain,3,sparse=1)
varianza_expl=pca_cvembedding.explained_variance_ratio_
fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()
ax1.bar(range(3),varianza_expl*100)
ax1.set_xlabel("Componente Principal",fontsize=14)
ax1.set_ylabel("Varianza de la componente principal",fontsize=14)
ax1.set_ylim(0,max(varianza_expl*100)*1.05)
#plt.show()
cum_variance=np.cumsum(varianza_expl)*100
ax2.plot(range(3), cum_variance,color="r")
ax2.set_ylim(0,100)
ax2.set_ylabel("Varianza acumulada", fontsize=14)
ax2.axhline(y=max(cum_variance),color="r",linestyle="--")
ax2.text(1,max(cum_variance)*0.9,f"Acumula {round(max(cum_variance),2)}%",fontsize=18)
plt.show()

#W2V
pcnumb=300
pca_w2vembedding,w2v_projections=get_pca(W2Vtrain,ytrain,pcnumb)
varianza_expl=pca_w2vembedding.explained_variance_ratio_
fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()
ax1.bar(range(pcnumb),varianza_expl*100)
ax1.set_xlabel("Componente Principal",fontsize=14)
ax1.set_ylabel("Varianza de la componente principal",fontsize=14)
ax1.set_ylim(0,max(varianza_expl*100)*1.05)
#plt.show()
cum_variance=np.cumsum(varianza_expl)*100
ax2.plot(range(pcnumb), cum_variance,color="r")
ax2.set_ylim(0,100)
ax2.set_ylabel("Varianza acumulada", fontsize=14)
ax2.axhline(y=max(cum_variance),color="r",linestyle="--")
ax2.text(pcnumb//2,max(cum_variance)*0.9,f"Acumula {round(max(cum_variance),2)}%",fontsize=18)
plt.show()

#dB
pcnumb=768
pca_dbembedding,db_projections=get_pca(Btrain,ytrain,pcnumb)
varianza_expl=pca_dbembedding.explained_variance_ratio_
fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()
ax1.bar(range(pcnumb),varianza_expl*100)
ax1.set_xlabel("Componente Principal",fontsize=14)
ax1.set_ylabel("Varianza de la componente principal",fontsize=14)
ax1.set_ylim(0,max(varianza_expl*100)*1.05)
#plt.show()
cum_variance=np.cumsum(varianza_expl)*100
ax2.plot(range(pcnumb), cum_variance,color="r")
ax2.set_ylim(0,100)
ax2.set_ylabel("Varianza acumulada", fontsize=14)
ax2.axhline(y=max(cum_variance),color="r",linestyle="--")
ax2.text(pcnumb//2,max(cum_variance)*0.9,f"Acumula {round(max(cum_variance),2)}%",fontsize=18)
plt.show()

#Graficamos las primeras 2 PC
fig, ax = plt.subplots(2, 3,sharex=True,sharey=True)
ax[0,0].scatter(cv_projections[:,0],cv_projections[:,1],c=ytrain,alpha=0.05)
ax[0,0].set_ylabel("PCA1")
ax[0,0].set_title("Bag of Words")
ax[1,0].scatter(cv_projections[:,0],cv_projections[:,2],c=ytrain,alpha=0.05)
ax[1,0].set_xlabel("PCA0")
ax[1,0].set_ylabel("PCA2")

ax[0,1].set_title("Word2Vec")
ax[0,1].scatter(w2v_projections[:,0],w2v_projections[:,1],c=ytrain,alpha=.05)
ax[1,1].scatter(w2v_projections[:,0],w2v_projections[:,2],c=ytrain,alpha=0.05)
ax[1,1].set_xlabel("PCA0")


ax[0,2].set_title("Distil BERT")
ax[0,2].scatter(db_projections[:,0],db_projections[:,1],c=ytrain,alpha=0.05)
ax[1,2].scatter(db_projections[:,0],db_projections[:,2],c=ytrain,alpha=0.05)
ax[1,2].set_xlabel("PCA0")
plt.show()



#pip install umap-learn
from umap import UMAP

neighs_to_test=[20,50,100,200]
fig, ax = plt.subplots(4, 3,sharex=True,sharey=True)
fig.set_figheight(15)
fig.set_figwidth(8)
for n,neigh in enumerate(neighs_to_test):
    umap_=UMAP(n_neighbors=neigh,n_components=2)
    umap_projs_cv=umap_.fit_transform(CVtrain,ytrain)
    umap_projs_w2v=umap_.fit_transform(W2Vtrain,ytrain)
    umap_projs_db=umap_.fit_transform(Btrain,ytrain)
    print(f"Neighbors {neigh}")
    if n==0:
        ax[n,0].set_title("Bag of Words")
        ax[n,1].set_title("Word2Vec")
        ax[n,2].set_title("Distil BERT")
    ax[n,0].scatter(umap_projs_cv[:,0],umap_projs_cv[:,1],c=ytrain,alpha=0.2)
    ax[n,1].scatter(umap_projs_w2v[:,0],umap_projs_w2v[:,1],c=ytrain,alpha=0.2)
    ax[n,2].scatter(umap_projs_db[:,0],umap_projs_db[:,1],c=ytrain,alpha=0.2)
    ax[n,0].set_ylabel(f"Neighbors: {neigh} \n UMAP1")
    if n==len(neighs_to_test):
        ax[n,0].set_xlabel("UMAP0")
        ax[n,1].set_xlabel("UMAP0")
        ax[n,2].set_xlabel("UMAP0")
plt.show()


np.argmax(pca_cvembedding.components_[0])



#C) 

'''
### **c) Pregunta de clasificación con métodos basados en árboles**

Otra pregunta interesante podría ser: **¿Podemos predecir si un recall será voluntario o mandatorio (`voluntary_mandated`) en función de las características del producto y el incidente?**

#### Pasos:
1. **Preprocesamiento:**
   - Similar al anterior, pero enfocado en la variable objetivo `voluntary_mandated`.

2. **Modelos basados en árboles:**
   - Árbol de decisión.
   - Random Forest.
   - Gradient Boosting (XGBoost, LightGBM, CatBoost).

3. **Evaluación:**
   - Métricas como precisión, recall, F1-score y matriz de confusión.
   - Importancia de características para entender qué variables influyen más en la predicción.
'''

'''
### **Ideas adicionales para el proyecto:**
- **Clustering:** Agrupar recalls similares basados en `reason_for_recall` y `product_description` usando técnicas como K-Means o DBSCAN.
- **Análisis de series temporales:** Predecir la cantidad de recalls en el futuro basado en datos históricos.
- **Detección de anomalías:** Identificar recalls que sean inusuales o atípicos en función de las características del dataset.
'''

#Para Bag of Words: Elegimos palabras frecuentes, analizamos su distribucion
#en los grupos y las listamos para ver si hay algo interesante.

CV_class1=CVtrain[ytrain==1]
CV_class2=CVtrain[ytrain==2]
CV_class3=CVtrain[ytrain==3]


def get_common_terms(cv_vector,threshold=100):
    sum_appearances=cv_vector.sum(axis=0)
    plt.bar(x=range(cv_vector.shape[1]),height=sum_appearances)
    plt.yscale("log")
    plt.axhline(threshold)
    plt.show()
    terms=[]
    above_treshold=np.argwhere(sum_appearances>threshold).T[0]
    features = countvectorizer.get_feature_names_out()
    for item in above_treshold:
        terms.append(features[item])
    return terms
terms_class1=get_common_terms(CV_class1)
len(terms_class3)
terms_class2=get_common_terms(CV_class2)
terms_class3=get_common_terms(CV_class3,20)
set1 = set(terms_class1)
set2 = set(terms_class2)
set3 = set(terms_class3)

# Palabras más probables de cada clase
venn3([set1, set2, set3], set_labels=('Clase I', 'Clase II', 'Clase III'))
plt.show()
diff1=(set1-set2)-set3
diff2=(set2-set1)-set3
diff3=(set3-set2)-set1

import math

def plot_words(words, title,color,row_height=1.0, max_rows=10):

    total_words = len(words)
    n_cols = math.ceil(total_words / max_rows)
    n_rows = min(max_rows, total_words)

    fig_width = 2 * n_cols  # Adjust for spacing
    fig_height = row_height * n_rows
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.title(title)
    # Plot each word at its (x, y) position
    for i, word in enumerate(words):
        col = i // max_rows
        row = i % max_rows
        x = col
        y = -row * row_height  # invert y to list top-down
        ax.text(x, y, word, va='top', ha='left', fontsize=12,color=color)

    # Set limits and hide axes
    ax.set_xlim(-0, n_cols)
    ax.set_ylim(-row_height * n_rows + 0, 0.2)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_words(diff1,"Clase I","r", row_height=0.30, max_rows=25)
plot_words(diff2,"Clase II","g", row_height=0.30, max_rows=25)
plot_words(diff3,"Clase III","b", row_height=0.30, max_rows=25)

#Para Word2Vec, hacemos PCA y KMEANS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

def pca_kmeans_analysis(data):
    # Step 1: Standardize the data (important for PCA and KMeans)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Step 2: Perform PCA
    pca = PCA()
    pca.fit(data_scaled)
    
    # Step 3: Plot explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.title('Explained Variance by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    plt.figure(figsize=(8, 6))    
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='x', linestyle='--', label='Cumulative Explained Variance')
    plt.title('Cummulative Variance by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cummulative Variance Ratio')
    plt.show()    
    # Ask for input on the number of components for PCA
    n_components = int(input("Enter the number of components for PCA: "))
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # Step 4: KMeans clustering (2 to 15 clusters)
    inertia = []
    silhouette_scores = []
    
    for n_clusters in range(2, 16):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data_pca)
        
        # Compute inertia (sum of squared distances to closest centroid)
        inertia.append(kmeans.inertia_)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(data_pca, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
    
    # Step 5: Elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 16), inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.show()
    
    # Step 6: Silhouette plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 16), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Ask for the optimal number of clusters based on the elbow and silhouette method
    optimal_k = int(input("Enter the desired number of clusters (based on elbow and silhouette method): "))
    
    # Perform final KMeans with the optimal number of clusters
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    final_kmeans.fit(data_pca)
    
    # Plot the clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=final_kmeans.labels_, cmap='viridis')
    plt.title(f'KMeans Clustering with {optimal_k} Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    # Plot the clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 2], c=final_kmeans.labels_, cmap='viridis')
    plt.title(f'KMeans Clustering with {optimal_k} Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    return pca, final_kmeans

pca_all_labels,cluster_all_labels=pca_kmeans_analysis(W2Vtrain)

pca_label1,clusters_label_1=pca_kmeans_analysis(W2Vtrain[ytrain==1])
pca_label2,clusters_label_2=pca_kmeans_analysis(W2Vtrain[ytrain==2])
pca_label3,clusters_label_3=pca_kmeans_analysis(W2Vtrain[ytrain==3])

#transform centroids into words:

clusters_original_space=pca_all_labels.inverse_transform(cluster_all_labels.cluster_centers_)
clusters_original_space1=pca_label1.inverse_transform(clusters_label_1.cluster_centers_)
clusters_original_space2=pca_label2.inverse_transform(clusters_label_2.cluster_centers_)
clusters_original_space3=pca_label3.inverse_transform(clusters_label_3.cluster_centers_)

def return_centroid_words(centroids):
    words=[]
    for centroid in centroids:
        # Assuming Word2Vec model maps centroids to word vectors
        # Find the closest word vector to the centroid
        closest_word = model.similar_by_vector(centroid, topn=1)[0][0]  # Getting the closest word
        words.append(closest_word)
    return words

palabras_centroide=return_centroid_words(clusters_original_space)
palabras_centroide1=return_centroid_words(clusters_original_space1)
palabras_centroide2=return_centroid_words(clusters_original_space2)
palabras_centroide3=return_centroid_words(clusters_original_space3)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import Counter

def pca_hierarchical_clustering(data: np.ndarray, n_components: int = 15,stride=5):
    if data.shape[0] < 100:
        raise ValueError("El dataset debe tener al menos 100 registros.")

    # PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    # Varianza explicada acumulada
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), varianza_acumulada, marker='o')
    plt.title('Varianza acumulada explicada por los componentes principales')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza acumulada explicada')
    plt.grid(True)
    plt.show()

    # Clustering jerárquico
    linkage_matrix = linkage(data_pca, method='ward')

    # Dendrograma truncado
    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix, no_labels=True, truncate_mode='level', p=10)
    plt.title('Dendrograma (truncado)')
    plt.xlabel('Observaciones')
    plt.ylabel('Distancia')
    plt.show()

    # Selección de número de clusters
    max_clusters = data.shape[0] // 100
   
    max_nmi=0
    for i in range(5,max_clusters,stride):
        cluster_labels = fcluster(linkage_matrix, t=i, criterion='maxclust')
        counts = Counter(cluster_labels)
        nmi=normalized_mutual_info_score(ytrain, cluster_labels)
        if nmi>max_nmi:
            max_n=i
            max_nmi=nmi
        print(i,nmi)
    n_clusters=max_n        
    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    counts = Counter(cluster_labels)

    # Calcular centroides en espacio PCA
    centroids = np.array([
        data_pca[cluster_labels == k].mean(axis=0)
        for k in range(1, n_clusters + 1)
    ])
        
    # Gráfico de los primeros 2 componentes con centroides
    plt.figure(figsize=(8, 6))
    for k in range(1, n_clusters + 1):
        plt.scatter(
            data_pca[cluster_labels == k, 0],
            data_pca[cluster_labels == k, 1],
            label=f'Cluster {k}',
            alpha=0.5
        )
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='red', marker='*', s=250, label='Centroides'
    )
    plt.title('Clusters en espacio PCA (2 componentes principales)')
    plt.xlabel('PC0')
    plt.ylabel('PC1')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    centroids=np.array([pca.inverse_transform(i) for i in centroids])

    return cluster_labels, data_pca, centroids


labels, reduced_data, centroids = pca_hierarchical_clustering(W2Vtrain,stride=1)

return_centroid_words(centroids)
pd.Series(labels).value_counts()
centroids.shape

#Hacemos TSNE para proyectar en baja dimensionalidad los embeddings word2vec y distilBERT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def visualizar_embeddings(embeddings, labels, metodo='all', random_state=42,perp=30,neigh=15):
    assert metodo in {'pca', 'tsne', 'umap', 'all'}, "Método no reconocido."

    def plot_2d(data_2d, title):
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, alpha=0.7)
        plt.title(f'{title}')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.colorbar(scatter, label='Etiqueta')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if metodo in {'pca', 'all'}:
        pca = PCA(n_components=2, random_state=random_state)
        data_pca = pca.fit_transform(embeddings)
        plot_2d(data_pca, 'PCA')

    if metodo in {'tsne', 'all'}:
        tsne = TSNE(n_components=2, random_state=random_state, 
                    perplexity=perp, init='pca')
        data_tsne = tsne.fit_transform(embeddings)
        plot_2d(data_tsne, 't-SNE')

    if metodo in {'umap', 'all'}:
        reducer = umap.UMAP(n_components=2, n_neighbors=neigh, random_state=random_state)
        data_umap = reducer.fit_transform(embeddings)
        plot_2d(data_umap, 'UMAP')
    
visualizar_embeddings(W2Vtrain,ytrain,metodo="tsne",perp=10)
visualizar_embeddings(W2Vtrain,ytrain,metodo="tsne",perp=30)
visualizar_embeddings(W2Vtrain,ytrain,metodo="tsne",perp=150)
visualizar_embeddings(W2Vtrain,ytrain,metodo="tsne",perp=300)


visualizar_embeddings(Btrain,ytrain,metodo="pca")

visualizar_embeddings(Btrain,ytrain,metodo="tsne",perp=10)
visualizar_embeddings(Btrain,ytrain,metodo="tsne",perp=30)
visualizar_embeddings(Btrain,ytrain,metodo="tsne",perp=150)
visualizar_embeddings(Btrain,ytrain,metodo="tsne",perp=300)

visualizar_embeddings(Btrain,ytrain,metodo="umap",neigh=15)
visualizar_embeddings(Btrain,ytrain,metodo="umap",neigh=60)
visualizar_embeddings(Btrain,ytrain,metodo="umap",neigh=200)
visualizar_embeddings(Btrain,ytrain,metodo="umap",neigh=400)

#Matriz de distancias
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from matplotlib.patches import Rectangle

def plot_cosine_groups(vectors, max_dist=0.3, min_size=2, reorder=True, figsize=(10, 8), cmap="viridis"):
    vectors = np.asarray(vectors)
    n = len(vectors)

    # Distancia coseno y clustering jerárquico
    dist_matrix = cosine_distances(vectors)
    linkage_matrix = linkage(vectors, method="average", metric="cosine")

    # Asignar clusters usando un umbral de distancia
    cluster_ids = fcluster(linkage_matrix, t=max_dist, criterion='distance')

    # Filtrar grupos pequeños
    valid_groups = {}
    for idx, cid in enumerate(cluster_ids):
        valid_groups.setdefault(cid, []).append(idx)
    valid_groups = {cid: idxs for cid, idxs in valid_groups.items() if len(idxs) >= min_size}

    if not valid_groups:
        print("⚠️ No se encontraron grupos que cumplan con los criterios.")
        return np.array([])

    # Reordenar para visualización
    if reorder:
        order = leaves_list(linkage_matrix)
    else:
        order = np.arange(n)
    reordered_dist = dist_matrix[order][:, order]
    cluster_ids_ordered = cluster_ids[order]
    labels = [f"v{i}" for i in order]

    # Mostrar heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(reordered_dist, xticklabels=labels, yticklabels=labels, cmap=cmap, square=True, cbar_kws={"label": "Distancia Coseno"})
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    plt.xlabel("Embeddings",size=20)
    plt.ylabel("Embeddings",size=20)
#    plt.title("Grupos por distancia máxima y tamaño mínimo")

    # Dibujar rectángulos sobre los grupos válidos
    color_map = plt.cm.tab10
    for i, (cid, members) in enumerate(valid_groups.items()):
        # Buscar posiciones de esos miembros en la matriz reordenada
        heatmap_positions = [np.where(order == m)[0][0] for m in members]
        min_pos, max_pos = min(heatmap_positions), max(heatmap_positions)
        rect = Rectangle((min_pos, min_pos), max_pos - min_pos + 1, max_pos - min_pos + 1,
                         linewidth=4, edgecolor=color_map(i % 10), facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

    # Calcular centroides
    centroides = []
    memb_number=[]
    for members in valid_groups.values():
        centroid = vectors[members].mean(axis=0)
        centroides.append(centroid)
        memb_number.append(len(members))
    return np.array(centroides),memb_number

grupos3,miembros3 = plot_cosine_groups(W2Vtrain[ytrain==3],max_dist=0.1,min_size=10,reorder=True)
palabras3=return_centroid_words(grupos3)
df=pd.DataFrame([palabras3,miembros3]).T
df.columns=["palabra","miembros"]
df.sort_values("miembros")

grupos2,miembros2 = plot_cosine_groups(W2Vtrain[ytrain==2],max_dist=0.1,min_size=10,reorder=True)
palabras2=return_centroid_words(grupos2)
df=pd.DataFrame([palabras2,miembros2]).T
df.columns=["palabra","miembros"]
df.sort_values("miembros").tail(6)

grupos1,miembros1 = plot_cosine_groups(W2Vtrain[ytrain==1],max_dist=0.1,min_size=10,reorder=True)
palabras1=return_centroid_words(grupos1)
df=pd.DataFrame([palabras1,miembros1]).T
df.columns=["palabra","miembros"]
df.sort_values("miembros").tail(6)