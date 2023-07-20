# AI-for-Science

solve_Burgers_Dirichlet.py：通过神经网络拟合偏微分方程（Burgers方程），使用Adam优化器和MSE损失函数对神经网络进行训练。参考：https://blog.csdn.net/lny161224/article/details/120520609

一组运行结果：

step: 0  loss = 0.4948219656944275

step: 1000  loss = 0.1647372543811798

step: 2000  loss = 0.15059393644332886

step: 3000  loss = 0.10797809064388275

step: 4000  loss = 0.0955720767378807

step: 5000  loss = 0.07511832565069199

step: 6000  loss = 0.02733888104557991

step: 7000  loss = 0.02113346941769123

step: 8000  loss = 0.017696112394332886

step: 9000  loss = 0.015311149880290031

t-SNE-digits.py：实现了使用 t-SNE（t-Distributed Stochastic Neighbor Embedding）算法对输入的经典手写数字数据集进行降维，并将结果可视化成二维散点图。

手写数字数据集包含了 1797 张 8x8 像素的手写数字图片和对应的标签。将图片数据 X 输入到 sklearn.manifold.TSNE 类中，使用 t-SNE 算法对其进行降维，得到二维的 X_tsne 数组。将 X_tsne 中的每个样本点可视化成二维散点图，其中每个点的颜色对应于其对应的手写数字标签 y。

![image](https://github.com/longziyu/AI-for-Science/blob/main/t-sne-digits.png)
