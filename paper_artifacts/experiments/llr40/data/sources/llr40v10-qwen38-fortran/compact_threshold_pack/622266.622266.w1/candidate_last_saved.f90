! Stream compaction: pack src[i]*weight[i] for every src[i] > 0.
! C-ABI: void compact_threshold_pack(double* src, double* weight, double* packed, int64_t* out_count, int64_t len)
subroutine compact_threshold_pack(src, weight, packed, out_count, len_1d) bind(C, name='compact_threshold_pack')
  use iso_c_binding, only: c_ptr, c_int64_t, c_double, c_f_pointer
  implicit none
  integer(c_int64_t), intent(in)    :: len_1d
  type(c_ptr), intent(in)           :: src, weight
  type(c_ptr), intent(inout)        :: packed, out_count
  real(c_double), dimension(:), pointer, contiguous :: s, w, p
  integer(c_int64_t), dimension(:), pointer, contiguous :: oc
  integer(c_int64_t) :: i, n

  call c_f_pointer(src, s, [len_1d])
  call c_f_pointer(weight, w, [len_1d])
  call c_f_pointer(packed, p, [len_1d])
  call c_f_pointer(out_count, oc, [1])

  n = 0
  do i = 1, len_1d
    if (s(i) > 0.0d0) then
      p(n + 1) = s(i) * w(i)
      n = n + 1
    end if
  end do
  oc(1) = n
end subroutine compact_threshold_pack
