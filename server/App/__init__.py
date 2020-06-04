from flask import Flask

from .ext import init_ext
from .settings import envs
from .views import init_blue


def create_app(env):

    app = Flask(__name__,static_url_path="/")

    # 初始化配置
    app.config.from_object(envs.get(env))

    # 初始化第三方
    init_ext(app)

    # 初始化路由
    init_blue(app)

    return app
