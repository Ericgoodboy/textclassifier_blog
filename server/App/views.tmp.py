from flask import Blueprint, request, render_template, make_response, Response, abort, session

blue = Blueprint('blue', __name__)


def init_blue(app):
    app.register_blueprint(blue)


@blue.route("/")
def index():
    return render_template('index.html')


@blue.route('/sendrequest/', methods=["GET","POST","DELETE", "PUT", "PATCH"])
def send_request():

    print(request.args)

    print(type(request.args))

    print(request.form)

    print(type(request.form))

    print(request.headers)

    print(type(request.headers))

    return 'send success'


@blue.route('/getresponse/')
def get_response():
    response = Response("自己造一个DIY")
    return response


@blue.errorhandler(401)
def handle_error(error):

    print(error)

    print(type(error))

    return 'What'


@blue.route('/login/', methods=["GET", "POST"])
def login():

    if request.method == "GET":
        return render_template('login.html')
    elif request.method == "POST":

        username = request.form.get("username")

        response = Response("登录成功%s" % username)

        # response.set_cookie('username', username)
        session['username'] = username
        session['password'] = "110"

        return response


@blue.route('/mine/')
def mine():

    # username = request.cookies.get('username')

    username = session.get('username')

    print(session)

    print(type(session))

    return '欢迎回来:%s' % username


@blue.route('/students/')
def students():

    student_list = ["小明 %d" % i for i in range(10)]

    return render_template('Students.html', student_list=student_list, a=5, b=5)


@blue.route("/userregister/")
def user_register():
    return render_template('user/user_register.html', title="用户注册")


@blue.route('/userregister2/')
def user_register2():

    users = ["小白用户%d" % i for i in range(10)]

    return render_template('user/user_register2.html', title="注册页面2", users=users, msg="hello")
