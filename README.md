# Curso de Machine Learning con Python

En este repositorio encontraras las prácticas correspondientes al Curso de Machine Learning con Python.

## Trabajo de investigación - Herramientas de python para machine learning.

<p>Realice un trabajo de investigación de las distintas herramientas que brinda Python para trabajar con machine learning separando por temas que cubre cada plataforma. En centrarse en sklearn y TensorFlow.</p>

## Frameworks and Libraries

<li> Librerias de Python para la Visualización</li>

<ul>
    <li> Matplotlib 🔥</li>
    <li> Seaborn 😎</li>
    <li> Bokeh😎 </li>
</ul>
<li>Librerías de Python para Cálculo Numérico y Análisis de Datos</li>
<ul>
    <li>NumPy 🖥</li>
    <li>SciPy 📝</li>
    <li>Pandas ⚡︎</li>
    <li>Numba 🎲</li>
</ul>

<li>Librerías de Python para Machine Learning</li>
<ul>
    <li>Scikit-learn 🖥</li>
</ul>
    <li>Librerías de Python para Deep Learning</li>
<ul>
    <li>TensorFlow 🖥</li>
    <li>Keras 🖥</li>
    <li>Pytorch 🖥</li>
</ul>
    <li>Librerías de Python para IA explicable</li>
<ul>
    <li>SHAP 🖥</li>
</ul>
    <li>Librerías de Python para Procesamiento de Lenguaje Natural</li>
<ul>
    <li>NLTK: Natural Language Toolkit</li>
    <li>Gensim </li>
    <li>Spacy </li>
</ul>

AI / ML ya no es una tecnología aspiracional sino una necesidad. Según Gartner, el 75% de todas las empresas que intentan implementar ML pondrán en práctica sus casos de uso para 2024.

# Machine Learning

<p>Python tiene un ecosistema increíble de bibliotecas que facilitan el inicio del aprendizaje automático: Scikit_learn (sklearn), Pandas y Matplotlib, etc. </p>

<p>Pandas: Es una de las librerías de python más útiles para los científicos de datos. Las estructuras de datos principales en pandas son Series para datos en una dimensión y DataFrame para datos en dos dimensiones.</p>

<p>Pandas es una librería de python que nos permite obtener datos de distinta procedencia como html, csv, json, pickle, sql. En el transcurso de las unidades iremos viendo como implementar pandas​ .Un ejemplo simple de cómo cargar un archivo csv podría ser:</p>

<p>NumPy se utiliza para crear matrices multidimensionales. Se puede convertir fácilmente a marcos de datos Panda y viceversa. De hecho, cuando los marcos de datos panda se alimentan con algoritmos sklearn ml, se traducen internamente en matrices NumPy antes del procesamiento.</p>

<p> Scipy solía procesar y evaluar datos como el modo de búsqueda, sesgo / curtosis de la función de distribución de probabilidad, etc.</p>

<p>La biblioteca Pickle se utiliza para serializar un modelo probado (que luego se puede guardar como un archivo binario) para su uso posterior. Por ejemplo, consulte algoritmos-> Bosque aleatorio -> Problema de iris. Luego puede exponer su modelo entrenado a través de llamadas REST que el cliente puede invocar para predecir valores usando el modelo.</p>

<p> Sklearn.metrics proporciona confusión_matriz y clasificación_report para evaluar el modelo.</p>

<p>Sklearn.datasets proporciona conjuntos de datos de muestra para probar varios análisis y algoritmos ML.</p>

<p>Nltk para el procesamiento del lenguaje natural.</p>

<p> MatplotLib y Seaborn se utilizan para trazar varios gráficos para analizar los datos. Seaborn es básicamente una envoltura sobre Matplotlib que ayuda a crear gráficos matplotlib con un atractivo más estético y también proporciona algunos tipos de gráficos más. (Consulte la carpeta de datos para los cuadernos de Python del curso intensivo en ambos)</p>

..

#

# TensorFlow

(An Open Source Machine Learning Framework for Everyone)

##

![picture alt](https://i0.wp.com/www.jessicayung.com/wp-content/uploads/2016/12/tensorflow-logo.jpg?fit=225%2C225 "Title is optional")

[Documentación](https://www.tensorflow.org/ "Named link title")

## Introduccion a TensorFlow

<p>Tensor flow es una biblioteca desarrollada por Google que nos permite realizar Machine Learning, con esta biblioteca podemos crear redes neuronales utilizando grafos.</p>

<p>TensorFlow es un sistema de programación en el que representamos cálculos en forma de grafos. Los nodos en el grafo se llaman ops (abreviatura de operaciones). Una op tiene cero o más tensores, realiza algún cálculo, y produce cero o más tensores.</p>

<p>Un grafo de TensorFlow es una descripción de cálculos. Para calcular cualquier cosa dentro de TensorFlow, el grafo debe ser lanzado dentro de una sesión. La Sesión coloca las operaciones del grafo en los diferentes dispositivos, tales como CPU o GPU, y proporciona métodos para ejecutarlas.</p>

## Tensor

<p>Los programas TensorFlow utilizan una estructura de datos llamada tensor para representar todos los datos. Puede pensar en un tensor como una matriz o lista n-dimensional. Un tensor tiene un tipo estático y dimensiones dinámicas. Sólo se pueden pasar tensores entre nodos en el grafo de cálculo.</p>

#

# Scikit-Learn

##

![picture alt](https://techrocks.ru/wp-content/uploads/2018/10/Scikit-learn.png "Title is optional")

[Documentación](https://scikit-learn.org/stable/getting_started.html "Named link title")

## Introduccion a Scikit-Learn

<p>Es una librería de python para Machine Learning y Análisis de Datos. Está basada en NumPy, SciPy y Matplotlib. La ventajas principales de scikit-learn son su facilidad de uso y la gran cantidad de técnicas de aprendizaje automático que implementa.</p>

<p>Con scikit-learn podemos realizar aprendizaje supervisado y no supervisado. Podemos usarlo para resolver problemas tanto de clasificación y como de regresión. </p>

[Link](https://www.iartificial.net/clasificacion-o-regresion/ "Named link title")

<p>Es muy fácil de usar porque tiene una interfaz simple y muy consistente. El interfaz es muy fácil de aprender. Te das cuenta que el interfaz es consistente cuando puedes cambiar de técnica de machine learning cambiando sólo una línea de código.</p>

<p>Suelen decir que es sencillo de usar porque tiene una interfaz simple y muy consistente. El interfaz es muy fácil de aprender. Te das cuenta que el interfaz es consistente cuando puedes cambiar de técnica de machine learning cambiando sólo una línea de código.</p>

<p>Otro punto a favor de scikit-learn es que los valores de los hiper-parámetros tienen unos valores por defecto adecuados para la mayoría de los casos.</p>

## Estas son algunas de las técnicas de aprendizaje automático que podemos usar con scikit-learn:

<ul>
    <li>Regresión Lineal y Polinómica.</li>
    <li>Regresión Logística.</li>
    <li>Máquinas de Vectores de Soporte.</li>
    <li>Arboles de Decisión.</li>
    <li>Bosques Aleatorios (Random Forests).</li>
    <li>Agrupamiento (Clustering).</li>
    <li>Modelos Basados en Instancias.</li>
    <li>Clasificadores Bayesianos.</li>
    <li>Reducción de Dimensionalidad.</li>
    <li>Detección de Anomalías.</li>
    <li>etc...</li>
</ul>

#

###### Fuentes

- [Wikipedia](https://es.wikipedia.org)
- [Tensorflow](https://www.tensorflow.org)
- [Scikit-Learn](https://es.wikipedia.org)
- [relopezbriega](http://relopezbriega.github.io/blog/2016/06/05/tensorflow-y-redes-neuronales/)
- [Herramientas en GNU/Linux para estudiantes universitarios](https://www.ibiblio.org/pub/linux/docs/LuCaS/Presentaciones/200304curso-glisa/redes_neuronales/curso-glisa-redes_neuronales-html/index.html)
- [Ejemplo tensorflow con iris dataset](http://tneal.org/post/tensorflow-iris/TensorFlowIris/)
- [Toptal](https://www.toptal.com/machine-learning/introducci%C3%B3n-a-la-teor%C3%ADa-de-aprendizaje-de-m%C3%A1quina-y-sus-aplicaciones-un-tutorial-visual-con-ejemplos/es)
