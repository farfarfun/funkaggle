import os
import zipfile

from notetool.download import download


def get_url(file_index=0):
    urls = [
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_00.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584504923&Signature=PZhFd0EWeG27T6Km3esWUsjASDUjw2uPiI7W%2BforX5krTX1T9iXuTDIr5hS6y87Fsdky0Y07RmHBSreWL0lf7jkOdxJwPxDU%2FCpmaHph00ZQSGg2SUcQw8IQXtqj%2FQ3t%2B4wRvsc7JvkhkfFy24PH8wQt2udqEh6MOcTFUL6MHofC%2BlZzhmQza6sgyWqDJLQlmPCMCvq%2ByzeYB5t4jvYknAO5w5%2FQvegfCnElcV0wCaufgfgJQjERgJ5RiTELOPh7dZZgMA7fAOHFrKoWX%2F7iXKVJ%2Fs9rXGmpFdv%2F02IGxQbEozGC%2F%2FQaT40zumAp3YmGnUgaktHJ2q6E5YlF6hOl%2BQ%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_01.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584802835&Signature=rL6pZnVoeCYAUPt5k%2BNjt7eQkGfdtk%2BrdvUunlK0Z5Ic4dhZUI%2FjimTUZNz4HgVwr3JGIkVpTRrhgsFMHMNxR28pmYzljPEUjQ89NpOvDwBt75%2FLxCrv3a5ySxJ%2Fz5%2F%2BGaD%2F3rLQZQ3Qn5Cg%2FgwLAGsaHAnxX%2FhhBglXQTeobP1KFg%2FY9YJkKvka64tHf6nd2KWOp6LqqGKyrvLBcU%2BcGfH2FUxGSdi9Je9ZuVYImq5bCvSiI0KtSOD3BueIrEimKl7bLLTKze5yUBzmeV%2B%2Fz37T7AURPatq2PEgL9UWuaBf6C6RGqom4Mq7mTVytYK7di7DBF2hub2wU1UNlW6iMQ%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_02.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584802849&Signature=M%2FY6evgbZU7sX%2B56gqJwxEP8n6Y4MmRg2EL8p2CPImopUau%2Fk%2B2hj6ea2ao25hlrZZgamdo1ccxSlT9JUyS8RsMZNGevlcXtasaTN34RU8dW456P9ipCNK2Zw%2FBD6bzZmy5mHseYC7iwdQAEjdAkOALWbm5Iz6nByB25P5o0cvETXrwfdxfZDTL9E4oHYnijwCDRKqp%2FMca1%2BvWnv%2B9aQYxtP%2BKkAYhICkLhYUytgRyv3H3cx68Kw4DaUy8sUagPBbmbPVG5bNKbA8pTlIC4w2CibvW3GF74Eg1X%2BUZ%2FP5hKRByEHu%2F1dNW5U3pnYovqAtUIsKLzW6RlglL9EtU%2Fpw%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_03.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584802854&Signature=Ik7tbpcGXCDowGmGYxWHduk%2B%2BzreiOxOz3mFWzuPicHXlO5J46dNUaTkZU8%2FBIZCCncqZ0M%2Fb%2BBcYKZIyW%2BaOKRwUXAaWHGXMOBLvibxySy1iNQXzYwO2JiJdgc5oNTrF9UZ%2BHqSFcbAtv4uA8GLrb62HsNAse49xmHTIdvirVIpWvxJsWJb59H7WrMw4Sae0yPh2VuNaaxQwwtJbgsnWilxcZXLPRPqIBnIOAn3xFR21cy%2BQtcbPkgBbTcMNMZt4F3OZPAXqqTCuf6BjyYfN5hctrdqTB08RZQnCewUS7bu%2FYUbl0mDClrxk4Vledinz82r%2FNY%2BOJLBSjdqYUuYiA%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_04.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584802859&Signature=EU1KEd6e9qkhQXqziAvVdiWrWqv9WKTDFZMi0YEHUBjhuD%2F%2FN5HUKMNf37xmxqKVOwhzjS24SWh5Osmrv2prwOJ4F%2F6ru5j68GPaLEV%2BsgnqVju8pcA4G%2BltlMbXQcYK5vqmPHXc%2F9dagXtb1SiX2lqh2FAcAU5ecBnKuOOXtJwKvaDmQ7yjGdMmvzjcvQC8fUDgf3MTTGLKuQBOnf6hYLB4AkGOTrMOEPFuyDoSHuYD3fZXQixMVbuHB00FGWc9an3KXC%2Ffo3WOuynT6F4p4jQLO1c4tER3KRRsm2K7%2F%2BxZJdc8r9do%2BVbxf0ciuHpiJFozerlxu9Q%2FjWhI8vqD7w%3D%3D',

        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_05.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584803030&Signature=l95sshebkoBA6cT9OBJo2GpckaWYMl7kir80P8QS4v82JHvwg%2F59Jd%2BGH6vnWvNwJMzPjnhk%2BZXfOchT1vAcW7rau1L3lnkU6ZtSPVqtR%2F7YQs6Ll%2BMS4Fspx5zEgxM6t6fF5MZ1CRp5Lr8DSW41ASWs9bUOEBLYu4VJ6tRS3D7bGqcXkm2jYc5M%2BUsTUofkVUjNPKm5xGX4TMkLbTy3BknLM8G5DKM9pu2ciaWfDAYDTo3eRmUjCYg%2FsfdeChlxBj9UBVTz9sjyFsoDTSLt11NrKVz56fto7i5%2BF%2BevUL%2BfP37MVcZ6ypNVNh5hrce0YqBpX0G%2BTdrcTPY6Gjz4Gw%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_06.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584803036&Signature=dcGAf7Cu3RDWtcw9Ool2calLZ6K4RlSLnYVRDcB8rDBEJ434OdJRh7cFsuGM15RrCAOK7q0UCrN0pEtBgYlbPLHtoXl9arXFnI3OztKMco6KMX3kZ1EDaCVepk6%2FH68EL3qOGztReedecEw7AVIm41rd%2B47wBZ9EXs0UP4V0QRUOLQB9K8NjsEOtsEj1eGKzAcTOExKKV6dW%2Ff1qaph5V%2FLLPdHDi2sf8tKfmjfvFhTU%2FvsPVxjOClsPTCAbSCluLrt89XcvurtZmiy32ILOz0V%2F%2FZAGSFHJxOviqG6xRAG%2FZxUGN1anBuJ3QjzzhcStzhS4y52aG0YL8evnnyrSjQ%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_07.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584803038&Signature=gwtUQ4g6lsyG5V%2FLAZi3xEihHblg2ATLfmopr573IKfrNXeU254FpTvDFDkjsY4OMXkTgHREy3IzII8RB9BGCCljfNxmP6EElLSKd3AVq2pRLU7vb97wPA016Px3W1826SG%2FrpFZ5hJUls7TXIB%2FFSyhQa%2FAALjOw0vhJ7LJCjTjkHBW%2F3yts3UF5%2BkXT7biDcivh9XgV8AEiWyw3m4yXS%2BkW1xC0La5ivohx6UiUbbq2ckRNriVj6wWIlseZXcl0kgb6GCFVfFTm5UcEAAOh%2Fxaed%2FBD2KYAOM%2BwBEhLO1HdUw7hLWiJJG9FXDk7q%2F9gjSo2T8PMq4Frm3%2Fs6D3mA%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_08.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584803041&Signature=cortPMvquSHpjGWsRHLemNBMKpDuiLzQg2JnyuxRTob2l75y%2BOTie0rPQYpicM29Ax4RyFbEHmcNHsq1a1Yuij%2B46fAzjqceZj3XDfIzF08f9PDxxy0HQmHzTR3aW80wzHwOgcvBl6C8rp72%2FAMZKuf20BqHGLdh%2F7%2FUEz0pas1HnnDP8uYJ7r6r0JjSj2MACiW6g76TiPMPBvVhFQCl%2Fpp544WwyuvWnptvwFnmdNoq4Pgwf00F6OY8D%2FwlE80Mt7zflkEgdEp7EStqDFpwIfZc7MhCED34UZDD2Cn9BNM2R8lJzI30cv3MTiNnlroqUxzZEnKPwUa%2FwK%2BzkuG5jA%3D%3D',
        'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_part_09.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584803043&Signature=A1o0%2FaIEroOStWD%2BgV9cMz3FgUHPgXz%2BRQedykeYPzt%2Fh%2F4SG9tXJYDWVhuPlmjPHZOG4NlyKxaLDopMiYKnmYNI1bg%2B%2BC%2FyvMG67C8jQWbxY6%2Ff5B3z9W%2BtUea8WhOtiSBAqZjXd%2FzpJipIlSqdMvm7jNe0Fm9qR2zps9%2BW3iPe%2B9%2FLGMpdlq8tcAaM5i2VuocWGYRZfWRoLrneHzwQx8BVzOfxfij23m19yyWfv1ul2qo4TH9p0gQRvCHx0Ih%2FK3bxJEUJbLndeyqqMZPB3kyHFwd7iAifdKY%2FfoCB7F1FBY93Kk5sP3L6whrQv%2BJZBejIX9JCLMUO6iSa5LLIug%3D%3D',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',

    ]

    _url = urls[file_index]
    return _url


def download_file(save_dir='/root/dataset/deepfake', file_index=0):
    file_name = 'dfdc_train_part_{}.zip'.format(str(file_index).rjust(2, '0'))
    save_path = os.path.join(save_dir, file_name)
    url = get_url(file_index)
    download(url, save_path)
    # os.system('unzip -n {}'.format(save_path))

    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(save_dir)
