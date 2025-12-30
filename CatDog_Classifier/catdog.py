import tensorflow as tf
from keras import models, layers, callbacks ,Sequential , preprocessing,optimizers,regularizers
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from keras.optimizers import Adam

# ---------- 设置中文字体 ----------
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据增强 ----------
data_augmentation = Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])



class StopAtLoss(callbacks.Callback):
    def __init__(self, monitor='val_loss', threshold=0.2):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if current < self.threshold:
            print(f"\n {self.monitor}={current:.4f} < {self.threshold}，提前停止训练")
            self.model.stop_training = True

            
# ---------- 2 层卷积模型 ----------
def build_model_2conv(input_shape=(128,128,3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(1e-05)),
        layers.Dropout(0.2), 
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer= Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_datasets(train_path, val_path, img_size=(128, 128), batch_size=64):
    # 创建原始训练数据集
    train_ds = preprocessing.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    # 创建验证数据集（不进行数据增强）
    val_ds = preprocessing.image_dataset_from_directory(
        val_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    print("类别顺序:", train_ds.class_names)

    # 应用数据增强到训练集
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 预取加速
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def train_model(train_path, val_path, epochs):
    model = build_model_2conv()
    train_ds, val_ds = create_datasets(train_path, val_path)

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )
    lr_callback = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath="cat_dog_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
    # stop_at_loss = StopAtLoss(monitor="val_loss", threshold=0.1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            lr_callback,
            checkpoint, 
            early_stop
            # stop_at_loss 
        ]
    )
    return history


def plot_history(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='训练 Loss')
    plt.plot(history.history['val_loss'], label='验证 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('值')
    plt.title('训练/验证 Loss & Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    train_path = '/home/jiangu/code/py/ML/catanddog/training_set'
    val_path = '/home/jiangu/code/py/ML/catanddog/test_set'
    history = train_model(train_path, val_path, epochs=100)
    plot_history(history)

if __name__ == '__main__':
    main()
