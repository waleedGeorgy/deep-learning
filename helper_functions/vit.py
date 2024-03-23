import torch
import torchvision
from torch import nn

# Class to create patch embeddings
class PatchEmbedding(nn.Module):
  """
  Transforms a 2D input image into a 1D vector of patch embeddings.

  Serves as data preparation for the ViT-Base/16 transformer.

  Args:
    - in_channels: (int) the number of channels in the input image (default = 3).
    - embedding_size: (int) the size of the flattened input embedding to the ViT (default = 768).
    - patch_size: (int) the size of patches created from the image (default = 16).
  """
  # Defining the contructor
  def __init__(self,
               in_channels: int = 3,
               embedding_size: int = 768,
               patch_size: int = 16):
    super().__init__()
    # The patch embedding creator sequential block
    self.patch_creator = nn.Sequential(
        # The convolutional layer the will create the patch embeddings
        nn.Conv2d(in_channels = in_channels,
                  out_channels = embedding_size,
                  kernel_size = patch_size,
                  stride = patch_size,
                  padding = 0),
        # The flattening layer that will flatten the patch embedding into a 1D vector
        nn.Flatten(start_dim = 2,
                   end_dim = 3)
    )

    self.patch_size = patch_size

  # Defining the forward method
  def forward(self, x):
    # Checking for image and patch size mismatching
    img_width = x.shape[-1]
    img_height = x.shape[-2]
    if img_height % self.patch_size == 0 and img_width % self.patch_size == 0:
      output_img = self.patch_creator(x)
      # Returning the permuted image to match the requirements of the ViT input
      return output_img.permute(0,2,1) # permute to (batch_size, patch_size, embedding_size)
    else:
      print(f'Could not create patches because the image size of ({img_height}x{img_width}) is not divisible by the patch size of {self.patch_size}')

# Class to create the ViT architecture
class ViT(nn.Module):
  def __init__(self,
               img_size:int = 224,
               img_channels: int = 3,
               embedding_size: int = 768,
               patch_size:int = 16,
               mlp_size:int = 3072,
               num_of_encoders:int = 12,
               num_of_msa_heads: int = 12,
               embedding_dropout: float = 0.1,
               msa_dropout: float = 0,
               mlp_dropout:float = 0.1,
               num_classes:int = 1000):
    """
    Creates the ViT-B/16 model architecture.
    Patchifies the input 2D image, adds classification embedding and position embedding,
    and passes the resulting 1D sequence of patches through multiple blocks of
    transformer encoders. Classification is performed by an MLP head.
    All the default values are found in the research paper.
    Args:
      img_size: Height and width of the input image (default = 224).
      img_channels: Number of channels in the input image (default = 3).
      embedding_size: Size of the embedding dimension (default = 768).
      patch_size: Size of patches the input image will be transformed into (default = 16).
      mlp_size: Size of the hidden units in the MLP block (default = 3072).
      num_of_encoders: Number of transformer encoder blocks (default = 12)
      num_of_msa_heads: Number of MSA heads inside the transformer encoder (default = 12).
      embedding_dropout: Amount of dropout to be applied after adding the position embedding (default = 0.1)
      msa_dropout: Amount of dropout in the MSA block (default = 0)
      mlp_dropout: Amount of dropout in the MLP block (default = 0.1)
      num_classes: Number of classes in the classification problem. (default = 1000)
    """
    super().__init__()

    # Checking whether the image can be divided into patches
    assert img_size % patch_size == 0, f'Image size must be divisible by patch size'

    # Getting the number of patches
    self.num_of_patches = int((img_size * img_size) / patch_size**2)

    self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_size),
                                            requires_grad=True)

    # Create learnable position embedding
    self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_of_patches+1, embedding_size),
                                               requires_grad=True)

    # Position embedding dropout
    self.embedding_dropout = nn.Dropout(p = embedding_dropout)

    # Patch creator layer (creates image patches and adds class tokens and posistion embedding)
    self.patch_creator = PatchEmbedding(in_channels = img_channels,
                                                embedding_size = embedding_size,
                                                patch_size = patch_size)

    # Transformer encoder layer (V2 ADDITION)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model = embedding_size,
                                                                                                nhead = num_of_msa_heads,
                                                                                                dim_feedforward = mlp_size,
                                                                                                dropout = mlp_dropout,
                                                                                                activation = 'gelu',
                                                                                                batch_first = True,
                                                                                                norm_first = True),
                                                     num_layers = num_of_encoders)

    # The output MLP head
    self.classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape = embedding_size),
        nn.Linear(in_features=embedding_size,
                  out_features = num_classes)
    )

  # Overwriting the forward() method
  def forward(self, x):
    # Getting the image batch
    batch_size = x.shape[0]
    # Equation 1
    # Expanding the class token to the entire batch
    class_token = self.class_embedding.expand(batch_size, -1, -1)
    # Patchifying the image
    patched_img = self.patch_creator(x)
    # Adding the class token
    patched_img_class_emb = torch.cat((class_token, patched_img), dim = 1)
    # Adding the position embedding
    patched_img_class_emb_pos_emb = patched_img_class_emb + self.position_embedding
    # Applying position embedding dropout
    trans_enc_input = self.embedding_dropout(patched_img_class_emb_pos_emb)
    # Equation 2 & 3
    # Passing the embeddings through the transformer encoder
    trans_enc_output = self.transformer_encoder(trans_enc_input)
    # Equation 4
    # Passing the 0th index (the classification token) through the classifier head
    img_class = self.classifier(trans_enc_output[:, 0])

    return img_class
