# @Geo Ahn
# executed in colab
# referred to https://github.com/leventt/surat

import torch
from torch import nn
import numpy as np
from scipy.signal import savgol_filter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda')
OUTPUT_COUNT = 61   # blendshape count
class Model(nn.Module):
    def __init__(self, moodSize, filterMood=False):
        super(Model, self).__init__()

        self.formantAnalysis = nn.Sequential(
            nn.Conv2d(1, 72, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(72, 108, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(108, 162, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(162, 243, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(243, 256, (1, 2), (1, 2)),
            nn.LeakyReLU(),
        )

        self.moodLen = 16
        mood = np.random.normal(.0, 1., (moodSize, self.moodLen))
        if filterMood:
            mood = savgol_filter(mood, 129, 2, axis=0)
        self.mood = nn.Parameter(
            torch.from_numpy(mood).float().view(moodSize, self.moodLen).to(DEVICE),
            requires_grad=True
        )

        self.articulation = nn.Sequential(
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (4, 1), (4, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(256 + self.moodLen, 150),
            nn.Linear(150, OUTPUT_COUNT),
            nn.Tanh(),
        )

    def forward(self, inp, mood, moodIndex=0):
        out = self.formantAnalysis(inp) 
        if mood is not None:
            out = torch.cat(
                (
                    out,
                    mood.view(
                        mood.view(-1, self.moodLen).size()[0], self.moodLen, 1, 1
                    ).expand(out.size()[0], self.moodLen, 64, 1)
                ),
                dim=1
            ).view(-1, 256 + self.moodLen, 64, 1)
        else:
            out = torch.cat(
                (
                    out,
                    # torch.cat((
                    #     self.mood[moodIndex.chunk(chunks=1, dim=0)],
                    #     self.mood[(moodIndex + 1).chunk(chunks=1, dim=0)]
                    # ), dim=0).view(
                    #     out.size()[0], self.moodLen, 1, 1
                        # cat : i_th mood and i+1_th mood -> row
                        # dim_0 time 2 or dim_1 time 2
                    self.mood[moodIndex.chunk(chunks=1, dim=0)].view(
                      out.size()[0], self.moodLen, 1, 1
                    ).expand(out.size()[0], self.moodLen, 64, 1)
                ),
                dim=1
            ).view(-1, 256 + self.moodLen, 64, 1)
        out = self.articulation(out)
        out = self.output(out.view(-1, 256 + self.moodLen))
        return out.view(-1, OUTPUT_COUNT)

############################
import torchaudio
from importlib.machinery import SourceFileLoader
import os

lpc = SourceFileLoader(
    'lpc',
    os.path.join(
        './LPCTorch/lpctorch/lpc.py'
    )
).load_module()

INPUT_VALUES_PRECALC_PATH = os.path.join(".", 'inputValues.precalc')

class Data(Dataset):
    def __init__(self, transforms=None, shiftRandom=True, validationAudioPath=None):
        self.transforms = transforms
        self.preview = validationAudioPath is not None
        self.shiftRandom = shiftRandom and not self.preview
        self.count = None

        animFPS = 60   # live link face app

        if self.preview:
            inputSpeechPath = validationAudioPath
            # inputSpeechPath = os.path.join(ROOT_PATH, 'merge_audio.wav')
        self.waveform, self.sampleRate = torchaudio.load(inputSpeechPath)
        if self.sampleRate != 16000:
            self.waveform = torchaudio.transforms.Resample(self.sampleRate, 16000)(self.waveform).to(DEVICE) # add .to(DEVICE)
            self.sampleRate = 16000

        self.count = int(animFPS * (self.waveform.size()[1] / self.sampleRate))

        self.LPC = lpc.LPCCoefficients(
            self.sampleRate,
            .032,
            .5,
            order=31  # 32 - 1
        )

        if os.path.exists(INPUT_VALUES_PRECALC_PATH):
            self.inputValues = torch.load(INPUT_VALUES_PRECALC_PATH).to(DEVICE) # add .to(DEVICE)
        else:
            print('pre-calculating input values...')
            self.inputValues = torch.Tensor([]) 
            audioFrameLen = int(.016 * 16000 * (64 + 1))
            audioHalfFrameLen = int(audioFrameLen / 2.)
            for i in range(self.count):
                print('{}/{}'.format(i + 1, self.count))
                audioRoll = -1 * (int(self.waveform.size()[1] / self.count) - audioHalfFrameLen)
                audioIdxRoll = int(i * audioRoll)
                audioIdxRollPair = int((i + 1) * audioRoll)

                self.inputValues = torch.cat(
                    (
                        self.inputValues,
                        torch.cat(
                            (
                                self.LPC(
                                    torch.roll(self.waveform[0:1, :], audioIdxRoll, dims=0)[:, :audioFrameLen]
                                ).view(1, 1, 64, 32),
                                self.LPC(
                                    torch.roll(self.waveform[0:1, :], audioIdxRollPair, dims=0)[:, :audioFrameLen]
                                ).view(1, 1, 64, 32)
                            ),
                            dim=0,
                        ).view(2, 1, 64, 32)
                    ), dim=0
                ).view(-1, 1, 64, 32)
            self.inputValues = self.inputValues.view(-1, 2, 1, 64, 32)
            torch.save(self.inputValues, INPUT_VALUES_PRECALC_PATH)

    def __getitem__(self, i):
        if i < 0:  # for negative indexing
            i = self.count + i
        inputValue = self.inputValues[i]

        if self.preview:
            return (
                torch.Tensor([i]).long(),
                inputValue[0],
                torch.zeros((1, OUTPUT_COUNT))
            )
        
        targetValue = torch.from_numpy(np.append(
            np.load(
                os.path.join(
                    ROOT_PATH,
                    'target',
                    'target_weight_{0}.npy'.format(i)
                )
            ),
            np.load(
                os.path.join(
                    ROOT_PATH,
                   'target',
                    'target_weight_{0}.npy'.format(i+1)
                )
            )
        )).float().view(-1, OUTPUT_COUNT)

        return (
            torch.Tensor([i]).long(),
            inputValue
            # output values are assumed to have max of 2 and min of -2
            (targetValue) * .5
        )

    def __len__(self):
        if self.preview:
            return self.count
        return self.count - 1  # for pairs

##################################
batchSize = 16
dataSet = Data(validationAudioPath="./audio_6.wav")
dataLoader = DataLoader(
    dataset=dataSet,
    batch_size=batchSize
    # num_workers=2
)
# model = Model(dataSet.count, filterMood=False).to(DEVICE)
model = Model(2682).to(DEVICE)
model.load_state_dict(torch.load("./1100.pth")) # train epoch is 1100
model.eval()

import csv

idx = 0
for i, data, target in dataLoader :
    i = i.to(DEVICE)
    data = data.to(DEVICE)
    output = model(data, None, i)
    ## 
    print(output)
    print(output.size())
    ##
    output_narray = output.cpu().detach().numpy()
    np.save('./output/blendshape_{0}'.format(idx), output_narray)
    idx = idx + 1
