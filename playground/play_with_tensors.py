import torch
import torch.nn.functional as F

# Play with softmax
# Assumption: batch_size = 3, seq_len = 2 -> Scores of shape [3,2,1]
# The following results should be obtained per batch element:
# Element 1: (0.26894, 0.72106),  Element 2: (0.01799, 0.98201),    Element 3: (0.18343, 0.81757)
attention_scores = torch.tensor([[[4],[5]], [[7],[11]], [[0.5],[2]]])
attention_weights = F.softmax(attention_scores, dim=1)
print("Normalized attention weights:")
print(attention_weights)

# Assumption: batch_size = 2, seq_len = 2, gru_output_dim = 3
# For the first two elements in the batch, the following calculation should be done:
# 0.2 * (1,2,3) + 0.8 * (4,5,6) = (3.4, 4.4, 5.4)
# For the third element in the batch, the following calculation should be done:
# 0.45 * (7,8,9) + 0.55 * (10,11,12) = (8.65, 9.65, 10.65)
gru_output = torch.tensor([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
attention_weights = torch.tensor([[[0.2],[0.8]],[[0.2],[0.8]],[[0.45],[0.55]]])
print("Intermediate output after element-wise mulitplication:")
print(gru_output*attention_weights)
print("Alternative with mul():")
print(torch.mul(gru_output, attention_weights))
attention_out = (gru_output*attention_weights).sum(axis=1)
print("Output:")
print(attention_out)