import random
import re
import unicodedata

def filter_all(text):
    return replace_fw(
        filter_categories(
            filter_replies(
                filter_urls(filter_spaces_tabs(
                    text
        )))))

def filter_categories(text):
    """Remove article categories. Eg. `[問卦]`

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.
    """

    return re.sub("\[.+\]", "", text)

def filter_replies(text):
    """Remove reply labels. Eg. `Re:`

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.
    """

    return re.sub("Re\:", "", text)

def filter_spaces_tabs(text):
    """Remove spaces and tabs.

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.
    """

    return re.sub(" |\t", "", text)

def filter_urls(text):
    """Remove urls.

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.
    """

    # https://gist.github.com/gruber/8891611
    return re.sub(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""", "", text)

def replace_fw(text):
    """Replace full-width characters with half-width's one. Eg. `？` -> `?`

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.
    """

    return unicodedata.normalize('NFKC', text)

def check_critical_words(text):
    """Check critical words defined in this function
       and determine whether the text will be used,
       according to given probabilities.

    Parameters
    ----------
    text : str
        Text to be checked.

    Returns
    -------
    bool
        True if the text will be used.
    """

    # Words with its corresponding occuring probability.
    words_prob = {
        "^推.{0,4}$": 0, # E.g. 推專業
        "^推.{5}": 0.7, # E.g. 推 這篇真的好棒棒棒棒
        "^蓋$": 0, # E.g. 蓋
        "^(一樓|二樓|三樓|四樓|五樓|樓上|樓下).{0,2}$": 0, # E.g. 五樓好臭
        "^(一樓|二樓|三樓|四樓|五樓|樓上|樓下).{3}": 0.2, # E.g. 五樓的屁眼
    }
    text = re.sub(' +', '', text)
    for key, value in words_prob.items():
        if re.match(f"{key}", text):
            if random.random() > value:
                return False
            break
    return True

if __name__ == "__main__":
    pass