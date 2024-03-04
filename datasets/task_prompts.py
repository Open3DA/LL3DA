TASK_PROPMT = {
    'densecap': [
        dict(
            instruction='### human: given the 3D scene, describe this object. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: describe this object in the given 3D scene. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe this object. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: localize and describe this object in the given 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, describe this object first, then localize it. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
        dict(
            instruction='### human: describe then localize the object from the 3D scene. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
    ],
    'ov-det': [
        dict(
            instruction='### human: what is this object? ### assistant:',
            answer='the {category} is localized at {locations}, {caption}',
        ),
    ],
    'qa': [
        dict(
            instruction='### human: given the 3D scene, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this quesiton according to the given 3D scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" with the related object locations in the input 3D scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the 3D scene, localize all the related objects first, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
}
BOX_FORMAT = '<obj>{}, {}, {}, {}, {}, {}</obj>'
COORD_FORMAT = '<loc>{}, {}</loc>'