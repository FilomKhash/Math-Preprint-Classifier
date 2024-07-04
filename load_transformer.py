# This script helps with loading the transformer model with custom layers trained in Math Archive Prediction Tasks with Transformers.ipynb whose weights were saved. 

import keras
from keras import layers
from keras import ops
import os


class LoadedTransformer:
    '''
    The trained model will be loaded into this class.
    '''
    def __init__(self,saved_weights_path='./models/clf_transformer.weights.h5'):
        
        assert os.path.exists(saved_weights_path), f"[Error] {saved_weights_path} does not exist."
        
        # Architecture parameters, DO NOT CHANGE. 
        embed_dim = 50            
        num_heads = 2             
        ff_dim = 64               
        vocabulary_size = 71614
        maxlen = 100
        
        # Dropout layers do not matter for inference, and hence are commented. 
        inputs = layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen, input_dim=vocabulary_size+1, embed_dim=embed_dim)
        x = embedding_layer(inputs)
        # x = layers.Dropout(0.1)(x)                                           
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim,
                                             # rate=0.2
                                            )
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Dense(200, activation="relu")(x)
        # x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(31, activation="softmax")(x)
        
        # Construct a model with this architecture via the Functional API. 
        model = keras.Model(inputs=inputs, outputs=outputs)
        # Load the saved weights. 
        model.load_weights(saved_weights_path)
        
        # Writing the model to an attribute. 
        LoadedTransformer._transformer=model
        
    
    # The getter
    @property
    def transformer(self):
        return self._transformer


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, input_dim, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=input_dim, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)