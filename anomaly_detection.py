import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

error_log = []

try:
    data = pd.read_csv('activity_log.csv', header=None, names=['timestamp', 'source', 'event_type', 'description', 'label'], on_bad_lines='skip')
except Exception as e:
    error_log.append(f"Ошибка при загрузке данных: {e}")

if 'data' in locals():
    try:
        data = data.drop(columns=['timestamp'])
        label_encoder = LabelEncoder()
        data['source'] = label_encoder.fit_transform(data['source'])
        data['event_type'] = label_encoder.fit_transform(data['event_type'])
        data['description'] = label_encoder.fit_transform(data['description'])
        data['label'] = label_encoder.fit_transform(data['label'])
    except Exception as e:
        error_log.append(f"Ошибка при препроцессинге данных: {e}")

    try:
        X = data.drop(columns=['label'])
        y = data['label']
    except Exception as e:
        error_log.append(f"Ошибка при выделении X и y: {e}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        error_log.append(f"Ошибка при разделении данных: {e}")

    for i in range(4):
        try:
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

            if history is not None:
                try:
                    plt.figure(figsize=(12, 5))

                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['loss'], label='loss')
                    plt.plot(history.history['val_loss'], label='val_loss')
                    plt.title(f'Loss over epochs (Run {i+1})')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['accuracy'], label='accuracy')
                    plt.plot(history.history['val_accuracy'], label='val_accuracy')
                    plt.title(f'Accuracy over epochs (Run {i+1})')
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy')
                    plt.legend()

                    plt.savefig(f'run_{i+1}_loss_accuracy_plot.png')  
                    plt.close()

                except Exception as e:
                    error_log.append(f"Ошибка при визуализации результатов в повторении {i+1}: {e}")

            try:
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                y_test = y_test.astype(int)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix (Run {i+1})')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'run_{i+1}_confusion_matrix.png')
                plt.close()

                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                report_df.to_csv(f'classification_report_run_{i+1}.csv')

            except Exception as e:
                error_log.append(f"Ошибка при оценке производительности в повторении {i+1}: {e}")

            try:
                model.save(f'anomaly_detection_model_run_{i+1}.h5')
            except Exception as e:
                error_log.append(f"Ошибка при сохранении модели в повторении {i+1}: {e}")

        except Exception as e:
            error_log.append(f"Ошибка при создании или обучении модели в повторении {i+1}: {e}")

if error_log:
    print("Произошли следующие ошибки:")
    for error in error_log:
        print(error)
