## HPLC_Matrix_Profil_dense_97percent_20220425012900.ipynb
- regressiós
- input első derivált
- konvoluciós
- 
![image](https://user-images.githubusercontent.com/30761411/165001423-0f6de0fc-7d96-47a1-8506-8d0941cbed01.png)

https://github.com/sipocz/LSTM_HPLC/blob/main/HPLC_Matrix_Profil_dense_97percent_20220425012900.ipynb


if _MODEL_=="convreg":
    
    from tensorflow.keras.layers import Input,Dense,Embedding,LSTM,TimeDistributed, Flatten, Bidirectional, Conv1D, MaxPooling1D, Dropout,Reshape,MaxPooling1D,Normalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adadelta,Adam,SGD,Adamax,RMSprop
    from tensorflow.keras.losses import sparse_categorical_crossentropy,categorical_crossentropy, mean_squared_error,binary_crossentropy

    K.clear_session
    x1=Input(shape=(max_input_length,1))
    x3=Conv1D(15,kernel_size=(10),kernel_regularizer="L2")(x1)
    x4=MaxPooling1D(3)(x3)
    x5=Conv1D(2,kernel_size=(2))(x4)
    x6=MaxPooling1D(3)(x5)


    #conv1=Conv1D(filters=8, kernel_size=2, padding='same', activation='relu')(embedded_x)
    #MP=MaxPooling1D(pool_size=1)(conv1)


    x7=Flatten()(x6)
    x7=Dropout(0.35)(x7)
    Dense_out= Dense(124, activation="selu",kernel_regularizer="L2")(x7) # 
    Dense_out=Dropout(0.5)(Dense_out)
    predictions= Dense(n_out, activation="selu",kernel_regularizer="L2")(Dense_out)
    model_convreg=Model(inputs=x1, outputs=predictions)
    
