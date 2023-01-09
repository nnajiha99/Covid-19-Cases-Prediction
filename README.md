# COVID-19 Cases Prediction

This project is intended to predict new cases of COVID-19 in Malaysia using the past 30 days of number of cases. For predicting the number of Covid-19 cases, LSTM architecture is used to construct the model. 
LSTM seems to perform better at discovering the trends of rising and dropping in this case.

The datasets were downloaded from https://github.com/MoH-Malaysia/covid19-public. The dataset folder contains the cases_malaysia_train.csv and cases_malaysia_test.csv. The following are some of the notable problems in the datasets:

    1. In train dataset, the data type of "cases_new" and "date" features are in object. Need to convert them into float. 
        The train dataset also has missing values. Interpolation has been used to remove the missing values.
    2. In test dataset,  the "cases_new" feature has missing values.

## Badges

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)




## Details of Steps

- Data Loading
    
    Load the cases_malaysia_train.csv dataset.

            CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv')
            train_df = pd.read_csv(CSV_PATH)
- Data Inspection

    Review the dataset for verication and debugging purposes before training the model.

            train_df.info() 
            train_df.duplicated().sum()
            train_df.isna().sum()
    
- Data Cleaning

    Remove the missing values on the dataset. Convert the data type for 'date' and 'cases_new' features.

            #Convert data type from object to float
            train_df['cases_new'] = pd.to_numeric(train_df['cases_new'], errors='coerce')
            train_df['date'] = pd.to_numeric(train_df['date'], errors='coerce')
            
            #Remove NaN using Interpolation
            train_df['cases_new'] = train_df['cases_new'].interpolate(method='polynomial',order=2)

- Features Selection

    In this project, we select the 'cases_new' feature.

            new_cases = train_df['cases_new'].values
    
- Data Preprocessing
    
    MinMaxScaler: Transform the feature by scaling feature to a given range.

    Window size is set to 30 days.

            mms = MinMaxScaler()
            new_cases = mms.fit_transform(new_cases[::,None])
            X = []
            y = []
            win_size = 30
            for i in range(win_size, len(new_cases)):
            X.append(new_cases[i-win_size:i])
            y.append(new_cases[i])

            X = np.array(X)
            y = np.array(y)

    Train-test-split

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=123)

- Model Development

    As informed above, LSTM is used for training the model. The nodes in the LSTM layers has been set as 64.

            model = Sequential()
            model.add(LSTM(64,return_sequences=True,input_shape=(X_train.shape[1:])))
            model.add(Dropout(0.3))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(64))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='relu'))

            model.summary()
            model.compile(optimizer='adam',loss='mse',metrics=['mse', 'mape'])

            plot_model(model, to_file='model.png')
    
            # Tensorboard callback
            LOGS_PATH = os.path.join(os.getcwd(), 'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
            tb = callbacks.TensorBoard(log_dir=LOGS_PATH)

            hist = model.fit(X_train,y_train,epochs=50,batch_size=64,callbacks=[tb])

    The training loss is displayed on TensorBoard.

            #For Google Colab 
            %load_ext tensorboard
            %tensorboard --logdir logs

- Model Analysis

    Load the cases_malaysia_test.csv dataset.

            TEST_CSV_PATH = os.path.join(os.getcwd(),'dataset', 'cases_malaysia_test.csv')
            test_df = pd.read_csv(TEST_CSV_PATH)

    Do data inspection/data cleaning

            test_df['date'] = pd.to_numeric(test_df['date'], errors='coerce')
            test_df.isna().sum()
            test_df['cases_new'] = test_df['cases_new'].interpolate(method='polynomial',order=2)

    Concatenate the cases_malaysia_test.csv dataset with cases_malaysia_train.csv dataset.

            concat = pd.concat((train_df['cases_new'],test_df['cases_new']))
            concat = concat[len(train_df['cases_new'])-win_size:]

    Min max transformation
            
            concat = mms.transform(concat[::,None])
    
            X_testtest = []
            y_testtest = []

            for i in range(win_size,len(concat)):
                X_testtest.append(concat[i-win_size:i])
                y_testtest.append(concat[i])

            X_testtest = np.array(X_testtest)
            y_testtest = np.array(y_testtest)

            predicted_cases = model.predict(X_testtest)

    ## Visualize the actual and predicted cases

            plt.figure()
            plt.plot(predicted_cases,color='red')
            plt.plot(y_testtest,color='blue')
            plt.legend(['Predicted Cases','Actual Cases'])
            plt.xlabel('Time')
            plt.ylabel('Covid-19 Cases Prediction')
            plt.show()


# Visualization

- Covid-19 Cases Prediction

![Covid-19_prediction](https://user-images.githubusercontent.com/121777112/211319099-8b476c98-30f6-476b-9ab0-93c7a6f1036f.png)

- Architecture of the model

![model](https://user-images.githubusercontent.com/121777112/211318555-dce7c52f-fada-4377-a83e-902d7183fe39.png)

- Training loss in TensorBoard

![epoch_loss](https://user-images.githubusercontent.com/121777112/211320250-bfb07ec0-3dc7-49bb-93f7-16506072720a.png)

![epoch_mape](https://user-images.githubusercontent.com/121777112/211324835-0de42bb8-4550-4fa7-a1f7-1e85cc41eb0b.png)
## Discussion

There is a weakness on this model where the Mean Absolute Percentage Error(MAPE) value is high. I executed the model.compile() function to add mape metrics to find out the MAPE. However when running the code, the mape at every epoch comes extremely huge and inconsistent. I have spent quite a long time trying to ensure the MAPE error is lesser than 1% and trying to improvise it.  
## Acknowledgements

 - GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.

