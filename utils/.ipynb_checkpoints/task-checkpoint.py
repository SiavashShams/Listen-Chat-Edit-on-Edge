
class TaskHandler2Speech2Audio():

    tasks = ['HE', 'HVC', 'OVC', 'RHVC', 'SE', 'SR', 'S↑', 'S↓', 
    'TAE', 'TAR', 'TA↑', 'TA↓', 'TSE', 'TSR', 'TS↑', 'TS↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target speech extraction
        if acts in ['1000', '0100']: # 2
            return 'TSE'
        # target audio extraction
        elif acts in ['0010', '0001']: # 2
            return 'TAE'
        # target speech removal
        elif acts in ['0111', '1011']: # 2
            return 'TSR'
        # target audio removal
        elif acts in ['1101', '1110']: # 2
            return 'TAR'
        # speech removal
        elif acts in ['0011']: # 1
            return 'SR'
        # speech enhancement
        elif acts in ['1100']: # 1
            return 'SE'
        # hybrid extraction
        elif acts in ['1001', '1010', '0101', '0110']: # 4
            return 'HE'
        # target speech volume amplification
        elif acts in ['U111', '1U11']: # 2
            return 'TS↑'
        # target speech volume reduction
        elif acts in ['D111', '1D11']: # 2
            return 'TS↓'
        # target audio volume amplification
        elif acts in ['11U1', '111U']: # 2
            return 'TA↑'
        # target audio volume reduction
        elif acts in ['11D1', '111D']: # 2
            return 'TA↓'
        # foreground amplification
        elif acts in ['UUDD', 'UU11', '11DD']: # 3
            return 'S↑'
        # background amplification
        elif acts in ['DDUU', '11UU', 'DD11']: # 3
            return 'S↓'
        # overall volume control
        elif acts in ['UUUU', 'DDDD']: # 2
            return 'OVC'
        # hybrid volume control
        elif '0' not in acts and ('U' in acts or 'D' in acts): # 64
            return 'HVC'
        # hybrid volume control + source removal
        elif '0' in acts and ('U' in acts or 'D' in acts): # 160
            return 'RHVC'
        # Silence
        elif acts in ['0000']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['1111']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TSE':
            return ['0100', '1000']
        elif task == 'TAE': 
            return ['0001', '0010']
        elif task == 'TSR': 
            return ['0111', '1011']
        elif task == 'TAR': 
            return ['1101', '1110']
        elif task == 'SR': 
            return ['0011']
        elif task == 'SE': 
            return ['1100']
        elif task == 'HE': 
            return ['0101', '0110', '1001', '1010']
        elif task == 'SILENCE':
            return ['0000']
        elif task == 'IDENTITY':
            return ['1111']
        elif task == 'TS↑':
            return ['1U11', 'U111']
        elif task == 'TS↓': 
            return ['D111', '1D11']
        elif task == 'TA↑': 
            return ['111U', '11U1']
        elif task == 'TA↓': 
            return ['11D1', '111D']
        elif task == 'S↑': 
            return ['UUDD', 'UU11', '11DD']
        elif task == 'S↓': 
            return ['DDUU', '11UU', 'DD11']
        elif task == 'OVC': 
            return ['DDDD', 'UUUU']
        elif task == 'HVC': 
            return ['DDD1','DDDU','DD1D',
            # 'DD11',
            'DD1U', 'DDUD', 'DDU1', 'D1DD',
            'D1D1', 'D1DU', 'D11D', 'D11U',
            'D1UD', 'D1U1', 'D1UU', 'DUDD',
            'DUD1', 'DUDU', 'DU1D', 'DU11',
            'DU1U', 'DUUD', 'DUU1', 'DUUU',
            '1DDD', '1DD1', '1DDU', '1D1D',
            '1D1U', '1DUD', '1DU1', '1DUU',
            # '11DD',
            '11DU', '11UD', '1UDD', '1UD1',
            '1UDU', '1U1D', '1U1U', '1UUD',
            '1UU1', '1UUU', 'UDDD', 'UDD1',
            'UDDU', 'UD1D', 'UD11', 'UD1U',
            'UDUD', 'UDU1', 'UDUU', 'U1DD',
            'U1D1', 'U1DU', 'U11D', 'U11U',
            'U1UD', 'U1U1', 'U1UU', 'UUD1',
            'UUDU', 'UU1D', 'UU1U', 'UUUD',
            'UUU1']
        elif task == 'RHVC': 
            return [
            '000D', '000U', '00D0','00DD',
            '00D1', '00DU', '001D', '001U',
            '00U0', '00UD', '00U1', '00UU',
            '0D00', '0D0D', '0D01', '0D0U',
            '0DD0', '0DDD', '0DD1', '0DDU',
            '0D10', '0D1D', '0D11', '0D1U',
            '0DU0', '0DUD', '0DU1', '0DUU',
            '010D', '010U', '01D0', '01DD',
            '01D1', '01DU', '011D', '011U',
            '01U0', '01UD', '01U1', '01UU',
            '0U00', '0U0D', '0U01', '0U0U',
            '0UD0', '0UDD', '0UD1', '0UDU',
            '0U10', '0U1D', '0U11', '0U1U',
            '0UU0', '0UUD', '0UU1', '0UUU',
            'D000', 'D00D', 'D001', 'D00U',
            'D0D0', 'D0DD', 'D0D1', 'D0DU',
            'D010', 'D01D', 'D011', 'D01U',
            'D0U0', 'D0UD', 'D0U1', 'D0UU',
            'DD00', 'DD0D', 'DD01', 'DD0U',
            'DDD0', 'DD10', 'DDU0', 'D100',
            'D10D', 'D101', 'D10U', 'D1D0',
            'D110', 'D1U0', 'DU00', 'DU0D',
            'DU01', 'DU0U', 'DUD0', 'DU10',
            'DUU0', '100D', '100U', '10D0',
            '10DD', '10D1', '10DU', '101D',
            '101U', '10U0', '10UD', '10U1',
            '10UU', '1D00', '1D0D', '1D01',
            '1D0U', '1DD0', '1D10', '1DU0',
            '110D', '110U', '11D0', '11U0',
            '1U00', '1U0D', '1U01', '1U0U',
            '1UD0', '1U10', '1UU0', 'U000',
            'U00D', 'U001', 'U00U', 'U0D0',
            'U0DD', 'U0D1', 'U0DU', 'U010',
            'U01D', 'U011', 'U01U', 'U0U0',
            'U0UD', 'U0U1', 'U0UU', 'UD00',
            'UD0D', 'UD01', 'UD0U', 'UDD0',
            'UD10', 'UDU0', 'U100', 'U10D',
            'U101', 'U10U', 'U1D0', 'U110',
            'U1U0', 'UU00', 'UU0D', 'UU01',
            'UU0U', 'UUD0', 'UU10', 'UUU0']
        else:
            raise ValueError(task)


class TaskHandler2Speech():

    tasks = ['HVC', 'OVC', 'RHVC', 'TSE', 'TS↑', 'TS↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target speech extraction
        if acts in ['01', '10']: # 2
            return 'TSE'
        # target speech volume amplification
        elif acts in ['U1', '1U']: # 2
            return 'TS↑'
        # target speech volume reduction
        elif acts in ['D1', '1D']: # 2
            return 'TS↓'
        # overall volume control
        elif acts in ['UU', 'DD']: # 2
            return 'OVC'
        # hybrid volume control
        elif acts in ['UD', 'DU']: # 2
            return 'HVC'
        # hybrid volume control + source removal
        elif acts in ['U0', '0U', 'D0', '0D']: # 4
            return 'RHVC'
        # Silence
        elif acts in ['00']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['11']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TSE':
            return ['01', '10']
        elif task == 'TS↑':
            return ['U1', '1U']
        elif task == 'TS↓': 
            return ['D1', '1D']
        elif task == 'OVC': 
            return ['DD', 'UU']
        elif task == 'HVC': 
            return ['UD', 'DU']
        elif task == 'RHVC': 
            return ['U0', '0U', 'D0', '0D']
        elif task == 'SILENCE':
            return ['00']
        elif task == 'IDENTITY':
            return ['11']
        else:
            raise ValueError(task)


class TaskHandler2Audio():

    tasks = ['HVC', 'OVC', 'RHVC', 'TAE', 'TA↑', 'TA↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target audio extraction
        if acts in ['01', '10']: # 2
            return 'TAE'
        # target audio volume amplification
        elif acts in ['U1', '1U']: # 2
            return 'TA↑'
        # target audio volume reduction
        elif acts in ['D1', '1D']: # 2
            return 'TA↓'
        # overall volume control
        elif acts in ['UU', 'DD']: # 2
            return 'OVC'
        # hybrid volume control
        elif acts in ['UD', 'DU']: # 2
            return 'HVC'
        # hybrid volume control + source removal
        elif acts in ['U0', '0U', 'D0', '0D']: # 4
            return 'RHVC'
        # Silence
        elif acts in ['00']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['11']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TAE':
            return ['01', '10']
        elif task == 'TA↑':
            return ['U1', '1U']
        elif task == 'TA↓': 
            return ['D1', '1D']
        elif task == 'OVC': 
            return ['DD', 'UU']
        elif task == 'HVC': 
            return ['UD', 'DU']
        elif task == 'RHVC': 
            return ['U0', '0U', 'D0', '0D']
        elif task == 'SILENCE':
            return ['00']
        elif task == 'IDENTITY':
            return ['11']
        else:
            raise ValueError(task)


class TaskHandler2Speech1Audio():

    tasks = ['HVC', 'OVC', 'RHVC', 'SE', 'SR', 'S↑', 'S↓', 'TSE', 'TSR', 'TS↑', 'TS↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target speech extraction
        if acts in ['010', '100']: # 2
            return 'TSE'
        # target speech removal
        if acts in ['011', '101']: # 2
            return 'TSR'
        # speech removal
        elif acts in ['001']: # 1
            return 'SR'
        # speech enhancement
        elif acts in ['110']: # 1
            return 'SE'
        # target speech volume amplification
        elif acts in ['1U1', 'U11']: # 2
            return 'TS↑'
        # target speech volume reduction
        elif acts in ['1D1', 'D11']: # 2
            return 'TS↓'
        # foreground amplification
        elif acts in ['11D', 'UU1', 'UUD']: # 3
            return 'S↑'
        # background amplification
        elif acts in ['11U', 'DD1', 'DDU']: # 3
            return 'S↓'
        # overall volume control
        elif acts in ['UUU', 'DDD']: # 2
            return 'OVC'
        # hybrid volume control
        elif '0' not in acts and ('U' in acts or 'D' in acts): # 14
            return 'HVC'
        # hybrid volume control + source removal
        elif '0' in acts and ('U' in acts or 'D' in acts): # 30
            return 'RHVC'
        # Silence
        elif acts in ['000']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['111']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TSE':
            return ['010', '100']
        elif task == 'TSR':
            return ['011', '101']
        elif task == 'SR':
            return ['001']
        elif task == 'SE':
            return ['110']
        elif task == 'TS↑':
            return ['1U1', 'U11']
        elif task == 'TS↓': 
            return ['1D1', 'D11']
        elif task == 'S↑':
            return ['11D', 'UU1', 'UUD']
        elif task == 'S↓': 
            return ['11U', 'DD1', 'DDU']
        elif task == 'OVC': 
            return ['UUU', 'DDD']
        elif task == 'HVC': 
            return ['1DD', '1DU', 
            '1UD', '1UU', 'D1D', 'D1U', 
            'DU1', 'DUD', 'DUU', 'U1D', 
            'U1U', 'UD1', 'UDD', 'UDU']
        elif task == 'RHVC': 
            return ['00D', '00U', 
            '01D', '01U', '0D0', '0D1', 
            '0DD', '0DU', '0U0', '0U1', 
            '0UD', '0UU', '10D', '10U', 
            '1D0', '1U0', 'D00', 'D01', 
            'D0D', 'D0U', 'D10', 'DD0', 
            'DU0', 'U00', 'U01', 'U0D', 
            'U0U', 'U10', 'UD0', 'UU0']
        elif task == 'SILENCE':
            return ['000']
        elif task == 'IDENTITY':
            return ['111']
        else:
            raise ValueError(task)


class TaskHandler1Speech2Audio():

    tasks = ['HVC', 'OVC', 'RHVC', 'SE', 'SR', 'S↑', 'S↓', 'TAE', 'TAR', 'TA↑', 'TA↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target audio extraction
        if acts in ['001', '010']: # 2
            return 'TAE'
        # target audio removal
        if acts in ['101', '110']: # 2
            return 'TAR'
        # speech removal
        elif acts in ['011']: # 1
            return 'SR'
        # speech enhancement
        elif acts in ['100']: # 1
            return 'SE'
        # target audio volume amplification
        elif acts in ['11U', '1U1']: # 2
            return 'TA↑'
        # target audio volume reduction
        elif acts in ['11D', '1D1']: # 2
            return 'TA↓'
        # foreground amplification
        elif acts in ['1DD', 'U11', 'UDD']: # 3
            return 'S↑'
        # background amplification
        elif acts in ['1UU', 'D11', 'DUU']: # 3
            return 'S↓'
        # overall volume control
        elif acts in ['UUU', 'DDD']: # 2
            return 'OVC'
        # hybrid volume control
        elif '0' not in acts and ('U' in acts or 'D' in acts): # 14
            return 'HVC'
        # hybrid volume control + source removal
        elif '0' in acts and ('U' in acts or 'D' in acts): # 30
            return 'RHVC'
        # Silence
        elif acts in ['000']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['111']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TAE':
            return ['001', '010']
        elif task == 'TAR':
            return ['101', '110']
        elif task == 'SR':
            return ['011']
        elif task == 'SE':
            return ['100']
        elif task == 'TA↑':
            return ['11U', '1U1']
        elif task == 'TA↓': 
            return ['11D', '1D1']
        elif task == 'S↑':
            return ['1DD', 'U11', 'UDD']
        elif task == 'S↓': 
            return ['1UU', 'D11', 'DUU']
        elif task == 'OVC': 
            return ['UUU', 'DDD']
        elif task == 'HVC': 
            return ['1DU', '1UD', 
            'D1D', 'D1U', 'DD1', 'DDU', 
            'DU1', 'DUD', 'U1D', 'U1U', 
            'UD1', 'UDU', 'UU1', 'UUD']
        elif task == 'RHVC': 
            return ['00D', '00U', 
            '01D', '01U', '0D0', '0D1', 
            '0DD', '0DU', '0U0', '0U1', 
            '0UD', '0UU', '10D', '10U', 
            '1D0', '1U0', 'D00', 'D01', 
            'D0D', 'D0U', 'D10', 'DD0', 
            'DU0', 'U00', 'U01', 'U0D', 
            'U0U', 'U10', 'UD0', 'UU0']
            
        elif task == 'SILENCE':
            return ['000']
        elif task == 'IDENTITY':
            return ['111']
        else:
            raise ValueError(task)


class TaskHandler3Speech1Audio():

    tasks = [
    'HE', 'HVC', 'OVC', 'RHVC', 
    'SE', 'SR', 'S↑', 'S↓', 
    'TSE', 'TSR', 'TS↑', 'TS↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target speech extraction
        if acts in ['0010', '0100', '1000']: # 3
            return 'TSE'
        # target speech removal
        elif acts in ['0111', '1011', '1101']: # 3
            return 'TSR'
        # speech removal
        elif acts in ['0001']: # 1
            return 'SR'
        # speech enhancement
        elif acts in ['1110']: # 1
            return 'SE'
        # hybrid extraction
        elif acts in ['0011', '0101', '0110', '1001', '1010', '1100']: # 6
            return 'HE'
        # target speech volume amplification
        elif acts in ['11U1', '1U11', 'U111']: # 3
            return 'TS↑'
        # target speech volume reduction
        elif acts in ['11D1', '1D11', 'D111']: # 3
            return 'TS↓'
        # foreground amplification
        elif acts in ['UUUD', 'UUU1', '111D']: # 3
            return 'S↑'
        # background amplification
        elif acts in ['DDDU', '111U', 'DDD1']: # 3
            return 'S↓'
        # overall volume control
        elif acts in ['UUUU', 'DDDD']: # 2
            return 'OVC'
        # hybrid volume control
        elif '0' not in acts and ('U' in acts or 'D' in acts): # 66
            return 'HVC'
        # hybrid volume control + source removal
        elif '0' in acts and ('U' in acts or 'D' in acts): # 160
            return 'RHVC'
        # Silence
        elif acts in ['0000']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['1111']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TSE':
            return ['0010', '0100', '1000']
        elif task == 'TSR': 
            return ['0111', '1011', '1101']
        elif task == 'SR': 
            return ['0001']
        elif task == 'SE': 
            return ['1110']
        elif task == 'HE': 
            return ['0011', '0101', '0110', '1001', '1010', '1100']
        elif task == 'SILENCE':
            return ['0000']
        elif task == 'IDENTITY':
            return ['1111']
        elif task == 'TS↑':
            return ['11U1', '1U11', 'U111']
        elif task == 'TS↓': 
            return ['11D1', '1D11', 'D111']
        elif task == 'S↑': 
            return ['UUUD', 'UUU1', '111D']
        elif task == 'S↓': 
            return ['DDDU', '111U', 'DDD1']
        elif task == 'OVC': 
            return ['DDDD', 'UUUU']
        elif task == 'HVC': 
            return ['11DD', '11DU', 
            '11UD', '11UU', '1D1D', '1D1U', 
            '1DD1', '1DDD', '1DDU', '1DU1', 
            '1DUD', '1DUU', '1U1D', '1U1U', 
            '1UD1', '1UDD', '1UDU', '1UU1', 
            '1UUD', '1UUU', 'D11D', 'D11U', 
            'D1D1', 'D1DD', 'D1DU', 'D1U1', 
            'D1UD', 'D1UU', 'DD11', 'DD1D', 
            'DD1U', 'DDU1', 'DDUD', 'DDUU', 
            'DU11', 'DU1D', 'DU1U', 'DUD1', 
            'DUDD', 'DUDU', 'DUU1', 'DUUD', 
            'DUUU', 'U11D', 'U11U', 'U1D1', 
            'U1DD', 'U1DU', 'U1U1', 'U1UD', 
            'U1UU', 'UD11', 'UD1D', 'UD1U', 
            'UDD1', 'UDDD', 'UDDU', 'UDU1', 
            'UDUD', 'UDUU', 'UU11', 'UU1D', 
            'UU1U', 'UUD1', 'UUDD', 'UUDU']
        elif task == 'RHVC': 
            return [
            '000D', '000U', '00D0','00DD',
            '00D1', '00DU', '001D', '001U',
            '00U0', '00UD', '00U1', '00UU',
            '0D00', '0D0D', '0D01', '0D0U',
            '0DD0', '0DDD', '0DD1', '0DDU',
            '0D10', '0D1D', '0D11', '0D1U',
            '0DU0', '0DUD', '0DU1', '0DUU',
            '010D', '010U', '01D0', '01DD',
            '01D1', '01DU', '011D', '011U',
            '01U0', '01UD', '01U1', '01UU',
            '0U00', '0U0D', '0U01', '0U0U',
            '0UD0', '0UDD', '0UD1', '0UDU',
            '0U10', '0U1D', '0U11', '0U1U',
            '0UU0', '0UUD', '0UU1', '0UUU',
            'D000', 'D00D', 'D001', 'D00U',
            'D0D0', 'D0DD', 'D0D1', 'D0DU',
            'D010', 'D01D', 'D011', 'D01U',
            'D0U0', 'D0UD', 'D0U1', 'D0UU',
            'DD00', 'DD0D', 'DD01', 'DD0U',
            'DDD0', 'DD10', 'DDU0', 'D100',
            'D10D', 'D101', 'D10U', 'D1D0',
            'D110', 'D1U0', 'DU00', 'DU0D',
            'DU01', 'DU0U', 'DUD0', 'DU10',
            'DUU0', '100D', '100U', '10D0',
            '10DD', '10D1', '10DU', '101D',
            '101U', '10U0', '10UD', '10U1',
            '10UU', '1D00', '1D0D', '1D01',
            '1D0U', '1DD0', '1D10', '1DU0',
            '110D', '110U', '11D0', '11U0',
            '1U00', '1U0D', '1U01', '1U0U',
            '1UD0', '1U10', '1UU0', 'U000',
            'U00D', 'U001', 'U00U', 'U0D0',
            'U0DD', 'U0D1', 'U0DU', 'U010',
            'U01D', 'U011', 'U01U', 'U0U0',
            'U0UD', 'U0U1', 'U0UU', 'UD00',
            'UD0D', 'UD01', 'UD0U', 'UDD0',
            'UD10', 'UDU0', 'U100', 'U10D',
            'U101', 'U10U', 'U1D0', 'U110',
            'U1U0', 'UU00', 'UU0D', 'UU01',
            'UU0U', 'UUD0', 'UU10', 'UUU0']
        else:
            raise ValueError(task)


class TaskHandler1Speech3Audio():

    tasks = [
    'HE', 'HVC', 'OVC', 'RHVC', 
    'SE', 'SR', 'S↑', 'S↓', 
    'TAE', 'TAR', 'TA↑', 'TA↓']
    tasks.sort()

    @staticmethod
    def group_task(acts):
        # target audio extraction
        if acts in ['0001', '0010', '0100']: # 3
            return 'TAE'
        # target audio removal
        elif acts in ['1011', '1101', '1110']: # 3
            return 'TAR'
        # speech removal
        elif acts in ['0111']: # 1
            return 'SR'
        # speech enhancement
        elif acts in ['1000']: # 1
            return 'SE'
        # hybrid extraction
        elif acts in ['0011', '0101', '0110', '1001', '1010', '1100']: # 6
            return 'HE'
        # target audio volume amplification
        elif acts in ['111U', '11U1', '1U11']: # 3
            return 'TA↑'
        # target audio volume reduction
        elif acts in ['111D', '11D1', '1D11']: # 3
            return 'TA↓'
        # foreground amplification
        elif acts in ['UDDD', 'U111', '1DDD']: # 3
            return 'S↑'
        # background amplification
        elif acts in ['DUUU', '1UUU', 'D111']: # 3
            return 'S↓'
        # overall volume control
        elif acts in ['UUUU', 'DDDD']: # 2
            return 'OVC'
        # hybrid volume control
        elif '0' not in acts and ('U' in acts or 'D' in acts): # 66
            return 'HVC'
        # hybrid volume control + source removal
        elif '0' in acts and ('U' in acts or 'D' in acts): # 160
            return 'RHVC'
        # Silence
        elif acts in ['0000']: # 1
            return 'SILENCE'
        # Identity
        elif acts in ['1111']: # 1
            return 'IDENTITY'
        else:
            raise ValueError(acts)

    @staticmethod
    def ungroup_task(task):
        if task == 'TAE':
            return ['0001', '0010', '0100']
        elif task == 'TAR': 
            return ['1011', '1101', '1110']
        elif task == 'SR': 
            return ['0111']
        elif task == 'SE': 
            return ['1000']
        elif task == 'HE': 
            return ['0011', '0101', '0110', '1001', '1010', '1100']
        elif task == 'SILENCE':
            return ['0000']
        elif task == 'IDENTITY':
            return ['1111']
        elif task == 'TA↑':
            return ['111U', '11U1', '1U11']
        elif task == 'TA↓': 
            return ['111D', '11D1', '1D11']
        elif task == 'S↑': 
            return ['UDDD', 'U111', '1DDD']
        elif task == 'S↓': 
            return ['DUUU', '1UUU', 'D111']
        elif task == 'OVC': 
            return ['DDDD', 'UUUU']
        elif task == 'HVC':
            return ['11DD', '11DU', 
            '11UD', '11UU', '1D1D', '1D1U', 
            '1DD1', '1DDU', '1DU1', '1DUD', 
            '1DUU', '1U1D', '1U1U', '1UD1', 
            '1UDD', '1UDU', '1UU1', '1UUD', 
            'D11D', 'D11U', 'D1D1', 'D1DD', 
            'D1DU', 'D1U1', 'D1UD', 'D1UU', 
            'DD11', 'DD1D', 'DD1U', 'DDD1', 
            'DDDU', 'DDU1', 'DDUD', 'DDUU', 
            'DU11', 'DU1D', 'DU1U', 'DUD1', 
            'DUDD', 'DUDU', 'DUU1', 'DUUD', 
            'U11D', 'U11U', 'U1D1', 'U1DD', 
            'U1DU', 'U1U1', 'U1UD', 'U1UU', 
            'UD11', 'UD1D', 'UD1U', 'UDD1', 
            'UDDU', 'UDU1', 'UDUD', 'UDUU', 
            'UU11', 'UU1D', 'UU1U', 'UUD1', 
            'UUDD', 'UUDU', 'UUU1', 'UUUD']
        elif task == 'RHVC': 
            return [
            '000D', '000U', '00D0','00DD',
            '00D1', '00DU', '001D', '001U',
            '00U0', '00UD', '00U1', '00UU',
            '0D00', '0D0D', '0D01', '0D0U',
            '0DD0', '0DDD', '0DD1', '0DDU',
            '0D10', '0D1D', '0D11', '0D1U',
            '0DU0', '0DUD', '0DU1', '0DUU',
            '010D', '010U', '01D0', '01DD',
            '01D1', '01DU', '011D', '011U',
            '01U0', '01UD', '01U1', '01UU',
            '0U00', '0U0D', '0U01', '0U0U',
            '0UD0', '0UDD', '0UD1', '0UDU',
            '0U10', '0U1D', '0U11', '0U1U',
            '0UU0', '0UUD', '0UU1', '0UUU',
            'D000', 'D00D', 'D001', 'D00U',
            'D0D0', 'D0DD', 'D0D1', 'D0DU',
            'D010', 'D01D', 'D011', 'D01U',
            'D0U0', 'D0UD', 'D0U1', 'D0UU',
            'DD00', 'DD0D', 'DD01', 'DD0U',
            'DDD0', 'DD10', 'DDU0', 'D100',
            'D10D', 'D101', 'D10U', 'D1D0',
            'D110', 'D1U0', 'DU00', 'DU0D',
            'DU01', 'DU0U', 'DUD0', 'DU10',
            'DUU0', '100D', '100U', '10D0',
            '10DD', '10D1', '10DU', '101D',
            '101U', '10U0', '10UD', '10U1',
            '10UU', '1D00', '1D0D', '1D01',
            '1D0U', '1DD0', '1D10', '1DU0',
            '110D', '110U', '11D0', '11U0',
            '1U00', '1U0D', '1U01', '1U0U',
            '1UD0', '1U10', '1UU0', 'U000',
            'U00D', 'U001', 'U00U', 'U0D0',
            'U0DD', 'U0D1', 'U0DU', 'U010',
            'U01D', 'U011', 'U01U', 'U0U0',
            'U0UD', 'U0U1', 'U0UU', 'UD00',
            'UD0D', 'UD01', 'UD0U', 'UDD0',
            'UD10', 'UDU0', 'U100', 'U10D',
            'U101', 'U10U', 'U1D0', 'U110',
            'U1U0', 'UU00', 'UU0D', 'UU01',
            'UU0U', 'UUD0', 'UU10', 'UUU0']
        else:
            raise ValueError(task)
