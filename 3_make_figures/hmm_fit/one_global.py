from one.api import ONE

# # the default one -- to use a first time to dowload data
# one = ONE()

# offline mode -- to use later on, once data have been downloaded
server = "alyx.internationalbrainlab.org"
cache_dir = f'../../../data/ONE/{server}'
one = ONE(cache_dir=cache_dir)


if __name__ == "__main__":

    print("ONE being used...")
    print(one)
    exit()
