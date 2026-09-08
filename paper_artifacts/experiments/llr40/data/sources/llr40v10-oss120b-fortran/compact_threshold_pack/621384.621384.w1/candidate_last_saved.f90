module compact_threshold_pack_mod_fp64
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, LEN_1D, workspace, workspace_bytes) bind(C, name="compact_threshold_pack_fp64")
    ! Arguments correspondence to C signature:
    !   const double *restrict src
    !   const double *restrict weight
    !   double *restrict packed
    !   int64_t *restrict out_count
    !   const int64_t LEN_1D
    real(c_double), intent(in) :: src(*)
    real(c_double), intent(in) :: weight(*)
    real(c_double), intent(out) :: packed(*)
    integer(c_int64_t), intent(out) :: out_count(*)
    integer(c_int64_t), value :: LEN_1D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes

    integer(c_int64_t) :: i, t
    integer(c_int64_t) :: nthreads
    integer(c_int64_t), allocatable :: count_per_thread(:)
    integer(c_int64_t), allocatable :: offset_per_thread(:)
    integer(c_int64_t) :: total
    integer(c_int64_t) :: local_idx

    if (LEN_1D <= 0_c_int64_t) then
        out_count(1) = 0_c_int64_t
        return
    end if

    nthreads = omp_get_max_threads()
    allocate(count_per_thread(nthreads))
    count_per_thread = 0_c_int64_t

    !$omp parallel private(t,i)
    t = omp_get_thread_num() + 1
    !$omp do schedule(static)
    do i = 1_c_int64_t, LEN_1D
        if (src(i) > 0.0_c_double) then
            count_per_thread(t) = count_per_thread(t) + 1_c_int64_t
        end if
    end do
    !$omp end do
    !$omp end parallel

    allocate(offset_per_thread(nthreads))
    offset_per_thread = 0_c_int64_t
    if (nthreads > 0) then
        offset_per_thread(1) = 0_c_int64_t
        do t = 2, nthreads
            offset_per_thread(t) = offset_per_thread(t-1) + count_per_thread(t-1)
        end do
    end if

    !$omp parallel private(t,i,local_idx)
    t = omp_get_thread_num() + 1
    local_idx = 0_c_int64_t
    !$omp do schedule(static)
    do i = 1_c_int64_t, LEN_1D
        if (src(i) > 0.0_c_double) then
            packed(offset_per_thread(t) + local_idx + 1) = src(i) * weight(i)
            local_idx = local_idx + 1_c_int64_t
        end if
    end do
    !$omp end do
    !$omp end parallel

    ! Compute total surviving count
    total = sum(count_per_thread)
    out_count(1) = total
    ! Zero out the tail of packed beyond total
    if (total < LEN_1D) then
        !$omp parallel do schedule(static)
        do i = total + 1, LEN_1D
            packed(i) = 0.0_c_double
        end do
        !$omp end parallel do
    end if
    deallocate(count_per_thread)
    deallocate(offset_per_thread)
  end subroutine compact_threshold_pack_fp64
end module compact_threshold_pack_mod_fp64
