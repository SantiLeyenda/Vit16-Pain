# Vit16-Pain
Repositorio de los archivos necesarios para correr el modelo y tambíen contiene el paper. 

Recomendaciones: 

- Antes de ver el codigo, es esencial entender el tesis de Alan.
- Entender el tesis en este repositorio 
- Entender las diferentes semanas y la diferencia entre pain y rest.
- Es importante aprender un poco de como funciona keras y tensorflow.
- Es imporante saber sobre inteligencia artificial, especialmente regresion lineal, CNNs, y transformers o vision transformers. 


  Como ejecutar el codigo:

1. Tener la computadora que utilizo Alan
2. Tener el kernel de GPU_tf2.10_py3.10(Python 3.10.0) puesto
3. Tener o crear una cuenta de WandB (Si es que se quiere usar sus funciones y cambiar el key puesto por el suyo)
4. Correr celda por celda de arriba hacia abajo
5. Tener el enviornment que sale en los archivos de este repositorio que se llama: enviroment.yml


Funcionamiento basico del codigo: 

1. Agarramos los datos o los files necesarios
2. En este caso solo utlizamos los momentos en los que las ratas estaban en dolor por lo cual eran 135 volumenes
3. Los pasamos o cargamos y dividimos en slices por que el vision transformer necesita imagenes en 2D
4. Entrenamos el modelo con estos datos
5. Obtuvimos resultados

Codigo mas a fondo: 

- En el codigo hay dos escenarios, uno en el cual es Male vs Male, y otro de Female vs Male
- El de Male vs Male esta comentado todo
- En el de Female vs Male solo esta comentado la parte que es diferente para que se entienda mejor la diferencia de que hacer en cada caso. 
