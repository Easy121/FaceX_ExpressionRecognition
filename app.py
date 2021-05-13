from utility.utility import *
from utility.visualizer import *


# 导入numpy压缩数据，列表类型
data = np.load('data/Abstract_NPZ/Female_front/happy.npz', allow_pickle=True)
# 可视化
vis = Visualizer(data[500])
# vis.visualizer_stroke()
vis.visualizer_stroke_gif()

# # x = [[1, 2, 3], [1, 2, 3]]
# # y = [[1, 2, 3], [3, 2, 1]]
# x = np.array([[1, 3], [1, 3]])
# y = np.array([[1, 3], [3, 1]])
# plt.plot(x, y)
# plt.show()

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)
#
# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,
#
# # animation function.  This is called sequentially
# def animate(i):
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     return line,
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)
#
# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.gif', fps=30)
#
# plt.show()
