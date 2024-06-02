import torch
import numpy as np
import io
from matplotlib import pyplot as plt

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

# @torch.jit.script
# def split_weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
#     num_lat = pred.shape[2]
#     #num_long = target.shape[2]
#     lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

#     # s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))

#     northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[441:])
#     southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:280])
#     tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[280:441])

#     northern_weight = torch.reshape(latitude_weighting_factor_torch(lat_t[441:], 280, northern_s), (1, 1, -1, 1))
#     southern_weight = torch.reshape(latitude_weighting_factor_torch(lat_t[:280], 280, southern_s), (1, 1, -1, 1))
#     tropics_weight = torch.reshape(latitude_weighting_factor_torch(lat_t[280:441], 161, tropics_s), (1, 1, -1, 1))



#     # weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
#     northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, 441:] - target[:, :, 441:])**2., dim=(-1,-2)))
#     southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :280] - target[:, :, :280])**2., dim=(-1,-2)))
#     tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, 280:441] - target[:, :, 280:441])**2., dim=(-1,-2)))
#     # result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
#     return northern_result, southern_result, tropics_result

# # @torch.jit.script
# # def split_weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
# #     #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
# #     num_lat = pred.shape[2]
# #     #num_long = target.shape[2]
# #     lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

# #     s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
# #     weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
# #     northern_result = torch.sqrt(torch.mean(weight[:,:,441:] * (pred[:, :, 441:] - target[:, :, 441:])**2., dim=(-1,-2)))
# #     southern_result = torch.sqrt(torch.mean(weight[:,:,:280] * (pred[:, :, :280] - target[:, :, :280])**2., dim=(-1,-2)))
# #     tropics_result = torch.sqrt(torch.mean(weight[:,:,280:441] * (pred[:, :, 280:441] - target[:, :, 280:441])**2., dim=(-1,-2)))
# #     # result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
# #     return northern_result, southern_result, tropics_result


# @torch.jit.script
# def split_weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     northern_result, southern_result, tropics_result = split_weighted_rmse_torch_channels(pred, target)
#     return torch.mean(northern_result, dim=0), torch.mean(southern_result, dim=0), torch.mean(tropics_result, dim=0)



# @torch.jit.script
def type_weighted_bias_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))

        result = torch.mean(weight * pred)

        # result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return result
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))


        northern_result = torch.mean(northern_weight * pred[:, :, northern_index:])

        # northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - clim_time_mean_daily[:, :, northern_index:])**2., dim=(-1,-2)))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))


        southern_result = torch.mean(southern_weight * pred[:, :, :souther_index])

        # southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - clim_time_mean_daily[:, :, :souther_index])**2., dim=(-1,-2)))
        return southern_result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))


        tropics_result = torch.mean(tropics_weight * pred[:, :, souther_index:northern_index])

        # tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - clim_time_mean_daily[:, :, souther_index:northern_index])**2., dim=(-1,-2)))
        return tropics_result
    else:
        raise NotImplementedError

# @torch.jit.script
def type_weighted_anomaly_torch_channels(pred: torch.Tensor, target: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))

        result_nume = torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) * (target - torch.mean(weight * target, dim=(-1, -2))))
        result_deno = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True))**2, dim=(-1, -2))) * torch.sqrt(torch.mean(weight * (target - torch.mean(weight * target, dim=(-1, -2), keepdim=True))**2, dim=(-1, -2)))

        result = result_nume / result_deno

        # result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return result
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))

        northern_result_nume = torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) * (target[:, :, northern_index:] - torch.mean(northern_weight * target[:, :, northern_index:], dim=(-1, -2))))
        northern_result_deno = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True))**2, dim=(-1, -2))) * torch.sqrt(torch.mean(northern_weight * (target[:, :, northern_index:] - torch.mean(northern_weight * target[:, :, northern_index:], dim=(-1, -2), keepdim=True))**2, dim=(-1, -2)))

        northern_result = northern_result_nume / northern_result_deno

        # northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - clim_time_mean_daily[:, :, northern_index:])**2., dim=(-1,-2)))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))


        southern_result_nume = torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) * (target[:, :, :souther_index] - torch.mean(southern_weight * target[:, :, :souther_index], dim=(-1, -2))))
        southern_result_deno = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True))**2, dim=(-1, -2))) * torch.sqrt(torch.mean(southern_weight * (target[:, :, :souther_index] - torch.mean(southern_weight * target[:, :, :souther_index], dim=(-1, -2), keepdim=True))**2, dim=(-1, -2)))

        southern_result = southern_result_nume / southern_result_deno

        # southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - clim_time_mean_daily[:, :, :souther_index])**2., dim=(-1,-2)))
        return southern_result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))


        tropics_result_nume = torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True)) * (target[:, :, souther_index:northern_index] - torch.mean(tropics_weight * target[:, :, souther_index:northern_index], dim=(-1, -2))))
        tropics_result_deno = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True))**2, dim=(-1, -2))) * torch.sqrt(torch.mean(tropics_weight * (target[:, :, souther_index:northern_index] - torch.mean(tropics_weight * target[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True))**2, dim=(-1, -2)))

        tropics_result = tropics_result_nume / tropics_result_deno

        # tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - clim_time_mean_daily[:, :, souther_index:northern_index])**2., dim=(-1,-2)))
        return tropics_result
    else:
        raise NotImplementedError



# @torch.jit.script
def type_weighted_activity_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))
        result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return result
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))
        northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - clim_time_mean_daily[:, :, northern_index:])**2., dim=(-1,-2)))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))
        southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - clim_time_mean_daily[:, :, :souther_index])**2., dim=(-1,-2)))
        return southern_result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))
        tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - clim_time_mean_daily[:, :, souther_index:northern_index])**2., dim=(-1,-2)))
        return tropics_result
    else:
        raise NotImplementedError


# @torch.jit.script
def type_weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))
        result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
        return result
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))
        northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - target[:, :, northern_index:])**2., dim=(-1,-2)))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))
        southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - target[:, :, :souther_index])**2., dim=(-1,-2)))
        return southern_result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))
        tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - target[:, :, souther_index:northern_index])**2., dim=(-1,-2)))
        return tropics_result
    else:
        raise NotImplementedError


# @torch.jit.script
def type_weighted_activity_torch(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_activity_torch_channels(pred, metric_type=metric_type)
    return torch.mean(result, dim=0)


# @torch.jit.script
def type_weighted_bias_torch(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_bias_torch_channels(pred, metric_type=metric_type)
    return torch.mean(result, dim=0)


# @torch.jit.script
def type_weighted_anomaly_torch(pred: torch.Tensor, target: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_anomaly_torch_channels(pred, target, metric_type=metric_type)
    return torch.mean(result, dim=0)

# @torch.jit.script
def type_weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_rmse_torch_channels(pred, target, metric_type=metric_type)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


# @torch.jit.script
def type_weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
        result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
        target, dim=(-1,-2)))
        return result
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))
        result = torch.sum(northern_weight * pred[:, :, northern_index:] * target[:, :, northern_index:], dim=(-1,-2)) / torch.sqrt(torch.sum(northern_weight * pred[:, :, northern_index:] * pred[:, :, northern_index:], dim=(-1,-2)) * torch.sum(northern_weight * target[:, :, northern_index:] *
        target[:, :, northern_index:], dim=(-1,-2)))
        return result                                             
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))
        result = torch.sum(southern_weight * pred[:, :, :souther_index] * target[:, :, :souther_index], dim=(-1,-2)) / torch.sqrt(torch.sum(southern_weight * pred[:, :, :souther_index] * pred[:, :, :souther_index], dim=(-1,-2)) * torch.sum(southern_weight * target[:, :, :souther_index] *
        target[:, :, :souther_index], dim=(-1,-2)))
        return result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))
        result = torch.sum(tropics_weight * pred[:, :, souther_index:northern_index] * target[:, :, souther_index:northern_index], dim=(-1,-2)) / torch.sqrt(torch.sum(tropics_weight * pred[:, :, souther_index:northern_index] * pred[:, :, souther_index:northern_index], dim=(-1,-2)) * torch.sum(tropics_weight * target[:, :, souther_index:northern_index] *
        target[:, :, souther_index:northern_index], dim=(-1,-2)))
        return result
    else:
        raise NotImplementedError



# @torch.jit.script
def type_weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_acc_torch_channels(pred, target, metric_type=metric_type)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

class Metrics(object):
    """
    Define metrics for evaluation, metrics include:

        - MSE, masked MSE;

        - RMSE, masked RMSE;

        - REL, masked REL;

        - MAE, masked MAE;

        - Threshold, masked threshold.
    """
    def __init__(self, epsilon = 1e-8, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(Metrics, self).__init__()
        self.epsilon = epsilon
    
    def MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth

        Returns
        -------

        The MSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()
    
    def Channel_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        channel_mse = torch.mean((pred - gt) ** 2, dim=[0,2,3])
        return channel_mse
    
    def Position_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        position_mse = torch.mean((pred - gt) ** 2, dim=[0, 1]).reshape(-1)
        return position_mse
    
    def RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        RMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The RMSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2, dim = [1, 2])
        return torch.mean(torch.sqrt(sample_mse)).item()
    
    def MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted

        gt: tensor, required, the ground-truth

        Returns
        -------
        
        The MAE metric.
        """
        sample_mae = torch.mean(torch.abs(pred - gt))
        return sample_mae.item()

    # def WRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
    #     """
    #     WRMSE metric.

    #     Parameters
    #     ----------

    #     pred: tensor, required, the predicted;

    #     gt: tensor, required, the ground-truth;


    #     Returns
    #     -------

    #     The WRMSE metric.
    #     """
    #     return weighted_rmse_torch(pred, gt)

    def Bias(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_bias_torch(pred - gt, metric_type="all") * data_std

    def NBias(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_bias_torch(pred - gt, metric_type="northern") * data_std

    def SBias(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_bias_torch(pred - gt, metric_type="southern") * data_std

    def TBias(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_bias_torch(pred - gt, metric_type="tropics") * data_std
    
    def Activity(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_activity_torch(pred - clim_time_mean_daily, metric_type="all") * data_std

    def NActivity(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_activity_torch(pred - clim_time_mean_daily, metric_type="northern") * data_std

    def SActivity(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_activity_torch(pred - clim_time_mean_daily, metric_type="southern") * data_std

    def TActivity(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_activity_torch(pred - clim_time_mean_daily, metric_type="tropics") * data_std

    def Anomaly(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_anomaly_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="all")

    def NAnomaly(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_anomaly_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="northern")

    def SAnomaly(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_anomaly_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="southern")

    def TAnomaly(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        return type_weighted_anomaly_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="tropics")


    def NWRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):

        return type_weighted_rmse_torch(pred, gt, metric_type="northern") * data_std
    

    def SWRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):

        return type_weighted_rmse_torch(pred, gt, metric_type="southern") * data_std
    

    def TWRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):

        return type_weighted_rmse_torch(pred, gt, metric_type="tropics") * data_std



    def WRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        WRMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The WRMSE metric.
        """

        return weighted_rmse_torch(pred, gt) * data_std

    # def WACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
    #     """
    #     WACC metric.

    #     Parameters
    #     ----------

    #     pred: tensor, required, the predicted;

    #     gt: tensor, required, the ground-truth;


    #     Returns
    #     -------

    #     The WACC metric.
    #     """
    #     return weighted_acc_torch(pred, gt)



    def NWACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):

        return type_weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="northern")
    

    def SWACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):

        return type_weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="southern")
    

    def TWACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):

        return type_weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily, metric_type="tropics")


    def WACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        WACC metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The WACC metric.
        """

        return weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)

class MetricsRecorder(object):
    """
    Metrics Recorder.
    """
    def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        metrics_list: list of str, required, the metrics name list used in the metric calcuation.

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(MetricsRecorder, self).__init__()
        self.epsilon = epsilon
        self.metrics = Metrics(epsilon = epsilon)
        self.metric_str_list = metrics_list
        self.metrics_list = []
        for metric in metrics_list:
            try:
                metric_func = getattr(self.metrics, metric)
                self.metrics_list.append([metric, metric_func, {}])
            except Exception:
                raise NotImplementedError('Invalid metric type.')
    
    def evaluate_batch(self, data_dict):
        """
        Evaluate a batch of the samples.

        Parameters
        ----------

        data_dict: pred and gt


        Returns
        -------

        The metrics dict.
        """
        pred = data_dict['pred']            # (B, C, H, W)
        gt = data_dict['gt']
        data_mask = None
        clim_time_mean_daily = None
        data_std = None
        if "clim_mean" in data_dict:
            clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
            data_std = data_dict["std"]

        losses = {}
        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            loss = metric_func(pred, gt, data_mask, clim_time_mean_daily, data_std)
            if isinstance(loss, torch.Tensor):
                for i in range(len(loss)):
                    losses[metric_name+str(i)] = loss[i].item()
            else:
                losses[metric_name] = loss

        return losses

    def plot_all_var(self, meters, metric_name):
        rmse = np.zeros(69)
        for i in range(69):
            rmse[i] = meters[metric_name+str(i)].global_avg

        fig = plt.figure(figsize=(15,15))
        singles = ['u10', 'v10', 't2m', 'msl']
        multis = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']

        plt.subplot(3,2,1)
        plt.scatter(range(4), rmse[:4], s=5)
        plt.xticks(range(4), singles, rotation = 30)
        plt.ylabel('RMSE')

        plt.subplot(3,2,2)
        plt.scatter(range(13), rmse[4:17], s=5)
        plt.xticks(range(13), multis, rotation = 30)
        plt.xlabel('z')
        plt.ylabel('RMSE')

        plt.subplot(3,2,3)
        plt.scatter(range(13), rmse[17:30], s=5)
        plt.xticks(range(13),multis,rotation = 30)
        plt.xlabel('q')
        plt.ylabel('RMSE')

        plt.subplot(3,2,4)
        plt.scatter(range(13), rmse[30:43], s=5)
        plt.xticks(range(13), multis, rotation = 30)
        plt.xlabel('u')
        plt.ylabel('RMSE')

        plt.subplot(3,2,5)
        plt.scatter(range(13), rmse[43:56], s=5)
        plt.xticks(range(13), multis, rotation = 30)
        plt.xlabel('v')
        plt.ylabel('RMSE')

        plt.subplot(3,2,6)
        plt.scatter(range(13), rmse[56:], s=5)
        plt.xticks(range(13),multis,rotation = 30)
        plt.xlabel('t')
        plt.ylabel('RMSE')

        plt.close()
        return fig
