from preprocessing import PreProcessing
from triplet_model import TripletLoss
import tensorflow as tf
import os

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
    processor = PreProcessing(data_src=data_path, TRIPLETS=100)
    triplet_model = TripletLoss()
    embedding_model = triplet_model.embedding()

    embedding_model(tf.random.normal([1, 75, 400, 3]))
    dataset = processor.get_triplets_batch().batch(2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(5):
        print(f"Epoch {epoch+1}")
        for anchor, positive, negative in dataset:
            with tf.GradientTape() as tape:
                anchor_out = embedding_model(anchor, training=True)
                positive_out = embedding_model(positive, training=True)
                negative_out = embedding_model(negative, training=True)

                pos_dist = tf.reduce_sum(tf.square(anchor_out - positive_out), axis=1)
                neg_dist = tf.reduce_sum(tf.square(anchor_out - negative_out), axis=1)
                loss = tf.maximum(pos_dist - neg_dist + 0.5, 0.0)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, embedding_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, embedding_model.trainable_variables))

        print(f"Epoch {epoch+1} loss: {loss.numpy():.4f}")

    save_path = os.path.join("models", "final.weights.h5")
    embedding_model.save_weights(save_path)
    print("Model saved to:", save_path)
